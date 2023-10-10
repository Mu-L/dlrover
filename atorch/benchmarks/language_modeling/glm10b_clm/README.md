# 下载wikitext-2-raw-v1数据集
https://huggingface.co/datasets/wikitext
```py
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk("/path/to/")
```

# 下载glm-10b
https://huggingface.co/THUDM/glm-10b

# 训练
```bash
bash launch_glm10b_training.sh
```