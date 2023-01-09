# Copyright 2022 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
from concurrent import futures
from typing import List

import grpc
from google.protobuf import empty_pb2

from dlrover.proto import elastic_training_pb2, elastic_training_pb2_grpc
from dlrover.python.common.constants import GRPC, NodeType, TrainingLoopStatus
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.elastic_training.elastic_ps import ElasticPsService
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.node_manager import NodeManager
from dlrover.python.master.shard.dataset_splitter import new_dataset_splitter
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.master.stats.training_metrics import OpStats, TensorStats
from dlrover.python.master.watcher.base_watcher import Node

_dlrover_context = Context.singleton_instance()
_DEFAULT_NUM_MINIBATCHES_PER_SHARD = 100


class MasterServicer(elastic_training_pb2_grpc.MasterServicer):
    """Master service implementation"""

    def __init__(
        self,
        task_manager: TaskManager,
        node_manager: NodeManager,
        speed_monitor: SpeedMonitor,
        rendezvous_server=None,
        job_metric_collector=None,
        elastic_ps_service=None,
    ):
        # TODO: group params together into a single object.
        self._task_manager = task_manager
        self._node_manager = node_manager
        self._speed_monitor = speed_monitor
        self._rendezvous_server = rendezvous_server
        self._job_metric_collector: JobMetricCollector = job_metric_collector
        self._elastic_ps_service: ElasticPsService = elastic_ps_service
        self._lock = threading.Lock()
        self._version = 0
        self._start_training_time = None
        self._start_autoscale = False

    def get_model_version(self):
        return self._version

    def report_task_result(self, request, _):
        success = True
        if request.err_message:
            logger.warning("Worker reported error: " + request.err_message)
            success = False
        task, _ = self._task_manager.report_dataset_task(request, success)
        if (
            not self._start_autoscale
            and self._node_manager
            and self._speed_monitor.completed_global_step == 0
            and int(time.time()) - self._start_training_time
            > _dlrover_context.seconds_to_autoscale_worker
        ):
            logger.info("Start autoscale for non-training jobs")
            self._node_manager.start_auto_scale()
            self._start_autoscale = True

        if (
            self._job_metric_collector
            and task.task_type == elastic_training_pb2.PREDICTION
        ):
            self._collect_runtime_stats()
            self._check_start_auto_scale_worker()
        return empty_pb2.Empty()

    def get_task(self, request, _):
        if not self._start_training_time:
            self._start_training_time = int(time.time())
        shard = elastic_training_pb2.Shard()
        res = elastic_training_pb2.Task(shard=shard)
        res.model_version = self._version
        ds_name = request.dataset_name
        dataset = self._task_manager.get_dataset(ds_name)
        if not dataset:
            return res
        task = self._task_manager.get_dataset_task(
            request.worker_type, request.worker_id, ds_name
        )

        if task:
            res.task_id = task.task_id
            res.type = task.task_type
            res.shard.name = task.shard.name
            res.shard.start = task.shard.start
            res.shard.end = task.shard.end
            res.shard.indices.extend(task.shard.record_indices)
        elif not dataset.completed():
            # If the todo and doing tasks are not empty,
            # Otherwise if the callback list is not empty,
            # we are trying to pop and invoke the callback.
            # Then the master tells the worker to wait
            # in case of new tasks later.
            if self._rendezvous_server:
                # If there is no more task, master only send wait task to
                # the last worker and other workers exit.
                if len(self._node_manager.get_running_workers()) == 1:
                    res.type = elastic_training_pb2.WAIT
            else:
                res.type = elastic_training_pb2.WAIT
        with self._lock:
            self._task_manager.reset_worker_start_task_time(request.worker_id)
        return res

    def report_dataset_shard_params(self, request, _):
        num_minibatches_per_task = (
            request.num_minibatches_per_shard
            or _DEFAULT_NUM_MINIBATCHES_PER_SHARD
        )
        shard_size = request.batch_size * num_minibatches_per_task
        splitter = new_dataset_splitter(
            request.shuffle,
            shard_size,
            request.dataset_size,
            request.num_epochs,
            request.dataset_name,
            request.storage_type,
        )
        self._task_manager.new_dataset(
            request.batch_size,
            request.dataset_size,
            request.dataset_name,
            splitter,
            request.task_type,
        )
        if self._job_metric_collector:
            self._job_metric_collector.collect_dataset_metric(
                request.dataset_name,
                request.dataset_size,
                request.storage_type,
            )
            if request.task_type == elastic_training_pb2.TRAINING:
                self._job_metric_collector.collect_training_hyper_params(
                    request.num_epochs, request.batch_size
                )
        return empty_pb2.Empty()

    def reset_sync(self, request, _):
        rendezvous_id = request.rendezvous_id
        worker_host = request.worker_host
        worker_local_process_id = request.worker_local_process_id
        res = elastic_training_pb2.DdpResetSyncResponse()
        res.reset = self._rendezvous_server.reset_sync(
            worker_host,
            worker_local_process_id,
            rendezvous_id,
        )
        return res

    def barrier_sync(self, request, _):
        rendezvous_id = request.rendezvous_id
        worker_host = request.worker_host
        worker_local_process_id = request.worker_local_process_id
        res = elastic_training_pb2.DdpInitSyncResponse()
        res.reset = self._rendezvous_server.barrier_sync(
            worker_host,
            worker_local_process_id,
            rendezvous_id,
        )
        return res

    def get_comm_rank(self, request, _):
        worker_host = request.worker_host
        worker_local_process_id = request.worker_local_process_id

        res = elastic_training_pb2.GetCommRankResponse()
        host_rank_info = self._rendezvous_server.get_worker_host_rank(
            worker_host, worker_local_process_id
        )
        if host_rank_info[0] == "horovod":
            res.rank_id = host_rank_info[1]
            res.local_rank = host_rank_info[2]
            res.local_size = host_rank_info[3]
            res.cross_rank = host_rank_info[4]
            res.cross_size = host_rank_info[5]
            res.rendezvous_port = self._rendezvous_server.get_rendezvous_port()
        elif host_rank_info[0] == "DDP":
            res.rank_id = host_rank_info[1]
            res.local_rank = host_rank_info[2]
            res.local_size = host_rank_info[3]
            res.master_addr = str(host_rank_info[4])
            res.master_port = host_rank_info[5]
        res.world_size = self._rendezvous_server.get_size()
        res.rendezvous_id = self._rendezvous_server.get_rendezvous_id()
        return res

    def report_training_loop_status(self, request, _):
        training_loop_status = request.status
        worker_local_process_id = request.worker_local_process_id
        worker_host = request.worker_host
        ddp_server_port = request.ddp_server_port

        if not self._rendezvous_server:
            logger.warning("The rendezvous server does not exit")
            return empty_pb2.Empty()

        logger.info(
            "Servicer get ddp_server_port:{} "
            "from host:{} id:{}".format(
                ddp_server_port, worker_host, worker_local_process_id
            )
        )
        if training_loop_status == TrainingLoopStatus.START:
            self._rendezvous_server.add_process(
                worker_host, worker_local_process_id, ddp_server_port
            )
        if training_loop_status == TrainingLoopStatus.END:
            self._rendezvous_server.remove_process(
                worker_host, worker_local_process_id
            )
        return empty_pb2.Empty()

    def ready_for_ps_relaunch(self, request, _):
        self._node_manager.post_ps_ready()
        return empty_pb2.Empty()

    def get_shard_checkpoint(self, request, _):
        res = elastic_training_pb2.ShardCheckpoint()
        dataset = self._task_manager.get_dataset(request.dataset_name)
        checkpoint = dataset.checkpoint()
        if checkpoint:
            res.content = checkpoint.to_json()
        else:
            res.content = ""
        return res

    def report_shard_checkpoint(self, request, _):
        res = elastic_training_pb2.ReportShardCheckpointResponse()
        success = self._task_manager.restore_dataset_from_checkpoint(
            request.content
        )
        res.success = success
        return res

    def report_used_resource(self, request, _):
        cpu = request.cpu
        memory = int(request.memory / 1024)  # Mi
        pod_id = request.node_id
        pod_type = request.node_type
        self._node_manager.update_node_resource_usage(
            pod_type, pod_id, cpu, memory
        )
        return empty_pb2.Empty()

    def get_dataset_epoch(self, request, _):
        res = elastic_training_pb2.GetDatasetEpochResponse()
        dataset_name = request.dataset_name
        res.epoch = self._task_manager.get_dataset_epoch(dataset_name)
        return res

    def report_model_metric(self, request, _):
        if self._job_metric_collector:
            tensor_stats = TensorStats(
                variable_count=request.tensor_stats.variable_count,
                total_variable_size=request.tensor_stats.total_variable_size,
                max_variable_size=request.tensor_stats.max_variable_size,
            )
            op_stats = OpStats(
                op_count=request.op_stats.op_count,
                update_op_count=request.op_stats.update_op_count,
                input_fetch_dur=request.op_stats.input_fetch_dur,
                flops=request.op_stats.flops,
            )
            self._job_metric_collector.collect_model_metric(
                tensor_stats,
                op_stats,
            )
        return empty_pb2.Empty()

    def report_global_step(self, request, _):
        self._speed_monitor.collect_global_step(
            request.global_step, request.timestamp
        )
        self._collect_runtime_stats()
        self._check_start_auto_scale_worker()
        return empty_pb2.Empty()

    def _collect_runtime_stats(self):
        if self._job_metric_collector:
            nodes = self._node_manager.get_running_nodes()
            self._job_metric_collector.collect_runtime_stats(
                self._speed_monitor, nodes
            )

    def _check_start_auto_scale_worker(self):
        sample_count = self._speed_monitor.get_sample_count()
        if (
            not self._start_autoscale
            and sample_count >= _dlrover_context.sample_count_to_adjust_worker
        ):
            logger.info(
                "Start autoscale with %s stats samples",
                sample_count,
            )
            self._node_manager.start_auto_scale()
            self._start_autoscale = True

    def get_cluster_version(self, request, _):
        response = elastic_training_pb2.GetClusterVersionResponse()
        if not self._elastic_ps_service:
            return response

        if request.task_type == NodeType.WORKER:
            response.version = self._elastic_ps_service.get_worker_version(
                request.version_type, request.task_id
            )
        elif request.task_type == NodeType.PS:
            response.version = self._elastic_ps_service.get_ps_version(
                request.version_type, request.task_id
            )
        return response

    def update_cluster_version(self, request, _):
        if not self._elastic_ps_service:
            return empty_pb2.Empty()

        if request.task_type == NodeType.WORKER:
            self._elastic_ps_service.update_worker_version(
                request.task_id, request.version_type, request.version
            )
        elif request.task_type == NodeType.PS:
            self._elastic_ps_service.update_ps_version(
                request.task_id, request.version_type, request.version
            )
        return empty_pb2.Empty()

    def query_ps_nodes(self, request, _):
        training_ps: List[Node] = self._node_manager.get_next_cluster_ps()
        ready = self._node_manager.ready_for_new_ps_cluster()
        res = elastic_training_pb2.QueryPsNodesResponse()
        for ps in training_ps:
            ps_meta = elastic_training_pb2.NodeMeta
            ps_meta.type = NodeType.PS
            ps_meta.addr = ps.service_addr
            ps_meta.cpu = ps.config_resource.cpu
            ps_meta.memory = ps.config_resource.memory
            res.ps_nodes.append(ps_meta)
        res.new_ps_ready = ready
        return res

    def query_running_nodes(self, request, _):
        nodes: List[Node] = self._node_manager.get_all_running_nodes()
        res = elastic_training_pb2.RunningNodes()
        for node in nodes:
            meta = elastic_training_pb2.NodeMeta()
            meta.type = node.type
            meta.addr = node.service_addr
            meta.cpu = node.config_resource.cpu
            meta.memory = node.config_resource.memory
            if node.config_resource.gpu_type:
                meta.gpu_type = node.config_resource.gpu_type
                meta.gpu_num = node.config_resource.gpu_num
            res.nodes.append(meta)
        return res

    def query_training_status(self, request, _):
        res = elastic_training_pb2.QueryTrainingStatusResponse()
        if self._task_manager.training_started():
            res.status = TrainingLoopStatus.START
        else:
            res.status = TrainingLoopStatus.PENDING
        return res

    def report_prestop(self, request, _):
        worker_host = request.worker_host
        self._rendezvous_server.report_prestop(worker_host)
        return empty_pb2.Empty()


def create_master_service(
    port,
    task_manager,
    node_manager,
    speed_monitor,
    rendezvous_server,
    job_metric_collector,
    elastic_ps_service,
) -> MasterServicer:
    """Create GRPC server"""
    logger.info("Creating master service")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=64),
        options=[
            ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
            (
                "grpc.max_receive_message_length",
                GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
            ),
        ],
    )
    master_servicer = MasterServicer(
        task_manager=task_manager,
        node_manager=node_manager,
        speed_monitor=speed_monitor,
        rendezvous_server=rendezvous_server,
        job_metric_collector=job_metric_collector,
        elastic_ps_service=elastic_ps_service,
    )

    elastic_training_pb2_grpc.add_MasterServicer_to_server(
        master_servicer, server
    )
    server.add_insecure_port("[::]:{}".format(port))
    logger.info("The port of the master server is: %d", port)
    return server
