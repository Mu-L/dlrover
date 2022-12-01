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

import copy
import threading
from typing import Dict

from dlrover.python.common.constants import (
    JobOptStage,
    NodeResourceLimit,
    NodeType,
    OptimizeWorkerPhase,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.resource.brain_optimizer import (
    BrainResoureOptimizer,
)
from dlrover.python.master.resource.local_optimizer import LocalOptimizer
from dlrover.python.master.resource.optimizer import ResourcePlan

_WORKER_OPTIMIZE_PHASE = "optimizer.worker.optimize-phase"

_dlrover_context = Context.instance()


def new_resource_optimizer(optimizer: str, job_uuid):
    if optimizer == BrainResoureOptimizer.name:
        return BrainResoureOptimizer(job_uuid)
    elif optimizer == LocalOptimizer.name:
        return LocalOptimizer(job_uuid)
    else:
        logger.error("Not support %s optimizer", optimizer)


class JobResource(object):
    def __init__(self):
        self.node_group_resources: Dict[str, NodeGroupResource] = {}

    def add_node_group_resource(
        self, node_type, num, resource_config, priority
    ):
        self.node_group_resources[node_type] = NodeGroupResource(
            count=num,
            node_resource=NodeResource.resource_str_to_node_resource(
                resource_config
            ),
            priority=priority,
        )

    def get_node_group_resource(self, node_type):
        return self.node_group_resources.get(node_type, None)

    def _get_group_node_num(self, node_type):
        if node_type in self.node_group_resources:
            return self.node_group_resources[node_type].count
        return 0

    def get_node_types(self):
        return list(self.node_group_resources.keys())

    def update_node_group_resource(self, node_type, num, cpu, memory):
        self.node_group_resources.setdefault(
            node_type,
            NodeGroupResource(
                count=0,
                node_resource=NodeResource(0, 0),
                priority=None,
            ),
        )
        resource = self.node_group_resources[node_type]
        resource.count = num or resource.count
        resource.node_resource.cpu = cpu or resource.node_resource.cpu
        resource.node_resource.memory = memory or resource.node_resource.memory

    @property
    def worker_num(self):
        return self._get_group_node_num(NodeType.WORKER)

    @property
    def ps_num(self):
        return self._get_group_node_num(NodeType.PS)

    @property
    def evaluator_num(self):
        return self._get_group_node_num(NodeType.EVALUATOR)

    @property
    def chief_num(self):
        return self._get_group_node_num(NodeType.CHIEF)

    def init_job_node_meta(
        self,
        relaunch_on_worker_failure,
        service_create_fn,
        new_node_name_fn,
    ):
        """
        job_resource: resource configuration of a job.
        relaunch_on_worker_failure: int, the number of relaunches.
        service_create_fn: a callable function to get the service address
            of a node.
        return: a dict with pod_type as key, and another dict as value.
                The other dict uses pod id as key, and PodInfo as value.
        """
        job_nodes: Dict[str, Dict[int, Node]] = {}
        for node_type in self.get_node_types():
            group_resource = self.get_node_group_resource(node_type)
            config_resource = group_resource.node_resource
            group_nodes: Dict[int, Node] = {}
            for i in range(group_resource.count):
                group_nodes[i] = Node(
                    node_type=node_type,
                    node_id=i,
                    name=new_node_name_fn(node_type, i),
                    config_resource=copy.deepcopy(config_resource),
                    max_relaunch_count=relaunch_on_worker_failure,
                    service_addr=service_create_fn(node_type, i),
                )
            job_nodes[node_type] = group_nodes
        return job_nodes


class JobResourceOptimizer(object):
    """It generates resource configuration for a job."""

    def __init__(
        self,
        worker_resource: NodeGroupResource,
        ps_resource: NodeGroupResource,
        optimizer: str,
        job_uuid="",
    ):
        self._worker_resource = worker_resource
        self._ps_resource = ps_resource
        self._original_worker_resource = copy.deepcopy(self._worker_resource)
        self._original_ps_resource = copy.deepcopy(self._ps_resource)
        self._resource_optimizer = new_resource_optimizer(optimizer, job_uuid)
        self._lock = threading.Lock()
        self.optimized_ps_mem = False
        self.optimize_worker_sampled = False
        self._job_stage = JobOptStage.CREATE

    def update_job_uuid(self, job_uuid):
        self._resource_optimizer.updaet_job_uuid(job_uuid)

    def _init_job_resource_by_optimizer(self):
        plan = self._resource_optimizer.generate_opt_plan(self._job_stage)
        if not plan or plan.empty():
            logger.info("Use the default plan to start the job")
            plan = ResourcePlan.new_default_plan()
        self._job_stage = JobOptStage.WORKER_INITIAL

        if (
            _dlrover_context.easydl_worker_enabled
            and NodeType.WORKER in plan.node_group_resources
        ):
            worker_resource = plan.node_group_resources[NodeType.WORKER]
            num, cpu, mem = self._check_ignore_original_worker_resource(
                worker_resource.count,
                worker_resource.node_resource.cpu,
                worker_resource.node_resource.memory,
            )
            self._worker_resource.update(num, cpu, mem)

        if (
            _dlrover_context.easydl_ps_enabled
            and NodeType.PS in plan.node_group_resources
        ):
            ps_resource = plan.node_group_resources[NodeType.PS]
            count, cpu, mem = self._check_ignore_original_ps_resource(
                ps_resource.count,
                ps_resource.node_resource.cpu,
                ps_resource.node_resource.memory,
            )
            self._ps_resource.update(count, cpu, mem)

    def optimize_worker_resource(self):
        plan = self._get_worker_resource_at_init_phase()
        if plan and NodeType.WORKER in plan.node_group_resources:
            worker_resource = plan.node_group_resources[NodeType.WORKER]
            self._worker_resource.update(
                worker_resource.count,
                worker_resource.node_resource.cpu,
                worker_resource.node_resource.memory,
            )

    def get_worker_resource(self):
        return self._worker_resource

    def init_job_resource(self, job_resource: JobResource):
        """Adjust the initial resource of typed pods by EasyDL.
        Args:
            job_resource: node resource configuration of a job.
        """
        self._init_job_resource_by_optimizer()
        job_resource.update_node_group_resource(
            NodeType.WORKER,
            self._worker_resource.count,
            self._worker_resource.node_resource.cpu,
            self._worker_resource.node_resource.memory,
        )

        job_resource.update_node_group_resource(
            NodeType.PS,
            self._ps_resource.count,
            self._ps_resource.node_resource.cpu,
            self._ps_resource.node_resource.memory,
        )

        evaluator_group = job_resource.get_node_group_resource(
            NodeType.EVALUATOR
        )
        if evaluator_group:
            resource = evaluator_group.node_resource
            if resource.cpu < NodeResourceLimit.MIN_VALID_CPU:
                resource.cpu = self._worker_resource.node_resource.cpu
            min_memory = NodeResourceLimit.MIN_VALID_MEMORY
            if resource.memory < min_memory:
                resource.memory = self._worker_resource.node_resource.memory

        return job_resource

    def adjust_oom_worker_resource(self, node: Node):
        """Increment the memory to launch worker. The new memory
        is max(1.5 * memory, the memory set by users).

        Args:
            node: Node object.
        """
        cur_mem = node.config_resource.memory
        if (
            _dlrover_context.easydl_worker_enabled
            and self._job_stage == JobOptStage.WORKER_INITIAL
        ):
            plan = self._resource_optimizer.generate_oom_recovery_plan(
                [node.name], JobOptStage.CREATE
            )
            if plan and not plan.empty():
                new_resource = plan.node_group_resources[NodeType.WORKER]
                self._worker_resource.node_resource.memory = max(
                    self._worker_resource.node_resource.memory,
                    new_resource.node_resource.memory,
                )
        else:
            self.optimize_worker_resource()
        cur_mem *= NodeResourceLimit.INCREMENTAL_MEMORY_FACTOR
        node.config_resource.memory = int(
            max(
                self._worker_resource.node_resource.memory,
                cur_mem,
                self._original_worker_resource.node_resource.memory,
            )
        )
        logger.info(
            "Increment the memory of %s to %s",
            node.name,
            node.config_resource.memory,
        )

    def adjust_oom_ps_resource(
        self, node: Node, training_started
    ) -> ResourcePlan:
        """Adjust PS resource if there is a OOM PS"""
        plan = self._resource_optimizer.generate_oom_recovery_plan(
            [node.name], JobOptStage.PS_INITIAL
        )
        if plan and not plan.empty():
            ps = plan.node_group_resources[NodeType.PS]
            if (
                not training_started
                and ps.count > 0
                and ps.node_resource.memory < NodeResourceLimit.MAX_MEMORY
            ):
                self._verify_optimized_group_resource(plan, NodeType.PS)
                plan.adjust_plan_by_context()
                return plan
            self._ps_resource.node_resource.memory = max(
                self._ps_resource.node_resource.memory,
                ps.node_resource.memory,
            )
        cur_mem = node.config_resource.memory
        cur_mem *= NodeResourceLimit.INCREMENTAL_MEMORY_FACTOR
        node.config_resource.memory = int(
            max(
                self._ps_resource.node_resource.memory,
                cur_mem,
                self._original_ps_resource.node_resource.memory,
            )
        )
        logger.info(
            "Increment the memory of %s to %s",
            node.name,
            node.config_resource.memory,
        )
        return ResourcePlan()

    def get_job_resource_plan(self):
        plan = None
        if self._job_stage == JobOptStage.WORKER_INITIAL:
            plan = self._get_worker_resource_at_init_phase()
            self._job_stage = JobOptStage.PS_INITIAL
        elif self._job_stage == JobOptStage.PS_INITIAL:
            plan = self._get_ps_resource_plan()
            self._job_stage = JobOptStage.RUNNING
        elif self._job_stage == JobOptStage.RUNNING:
            plan = self._get_ps_resource_plan()
            if plan.empty():
                plan = self._get_worker_resource_at_running()
        if plan.empty():
            return None

        if NodeType.WORKER in plan.node_group_resources:
            self._verify_optimized_group_resource(plan, NodeType.WORKER)

        if plan and NodeType.PS in plan.node_group_resources:
            self._verify_optimized_group_resource(plan, NodeType.PS)

        plan.adjust_plan_by_context()
        return plan

    def _get_worker_resource_at_running(self):
        if not self.optimize_worker_sampled:
            plan = self._get_worker_resource_at_sample_phase()
            self.optimize_worker_sampled = True
        else:
            plan = self._get_worker_resource_at_stable_phase()
        return plan

    def _get_worker_resource_at_init_phase(self):
        optimizer_config = {}
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.INITIAL
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if plan.empty():
            logger.info("No any plan to initialize the number of worker")
            return

        return plan

    def _get_worker_resource_at_sample_phase(self):
        optimizer_config = {}
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.SAMPLE
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if not plan:
            return
        return plan

    def _get_worker_resource_at_stable_phase(self):
        optimizer_config = {}
        optimizer_config[_WORKER_OPTIMIZE_PHASE] = OptimizeWorkerPhase.STABLE
        plan = self._resource_optimizer.generate_opt_plan(
            JobOptStage.WORKER_INITIAL, optimizer_config
        )
        if not plan:
            return
        return plan

    def _get_ps_resource_plan(self):
        optimizer_config = {}
        plan = self._resource_optimizer.generate_opt_plan(
            self._job_stage, optimizer_config
        )
        return plan

    def _verify_optimized_group_resource(self, plan: ResourcePlan, node_type):
        group = plan.node_group_resources[node_type]
        if node_type == NodeType.WORKER:
            num, cpu, mem = self._check_ignore_original_worker_resource(
                group.count,
                group.node_resource.cpu,
                group.node_resource.memory,
            )
            self._worker_resource.count = num
            self._worker_resource.node_resource.cpu = cpu
            self._worker_resource.node_resource.memory = mem
        elif node_type == NodeType.PS:
            num, cpu, mem = self._check_ignore_original_ps_resource(
                group.count,
                group.node_resource.cpu,
                group.node_resource.memory,
            )
            self._ps_resource.count = min(num, NodeResourceLimit.MAX_PS_NUM)
            self._ps_resource.node_resource.cpu = cpu
            self._ps_resource.node_resource.memory = mem
        group.count = num
        group.node_resource.cpu = cpu
        group.node_resource.memory = mem
        return group

    def _check_ignore_original_worker_resource(self, num, cpu, mem):
        """Abandon the optimization result if users have set the resource."""
        #  Users may worry about that the increasing number of worker hurts the
        #  accuracy, so the max number of worker is the configuration.
        if self._original_worker_resource.count > 0:
            num = self._original_worker_resource.count
        if (
            self._original_worker_resource.node_resource.memory
            >= NodeResourceLimit.MIN_VALID_MEMORY
        ):
            mem = self._original_worker_resource.node_resource.memory
        if (
            self._original_worker_resource.node_resource.cpu
            >= NodeResourceLimit.MIN_VALID_CPU
        ):
            cpu = self._original_worker_resource.node_resource.cpu
        return num, cpu, mem

    def _check_ignore_original_ps_resource(self, num, cpu, mem):
        """Abandon the optimization result if users have set the resource."""
        if self._original_ps_resource.count > 0:
            num = self._original_ps_resource.count
        if (
            self._original_ps_resource.node_resource.memory
            >= NodeResourceLimit.MIN_VALID_MEMORY
        ):
            mem = self._original_ps_resource.node_resource.memory
        if (
            self._original_ps_resource.node_resource.cpu
            >= NodeResourceLimit.MIN_VALID_CPU
        ):
            cpu = self._original_ps_resource.node_resource.cpu
        return num, cpu, mem