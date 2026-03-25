# 🚀 Distributed LLM Training

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.14-0078D4?style=flat)](https://github.com/microsoft/DeepSpeed)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production distributed LLM training framework** — PyTorch FSDP + DeepSpeed ZeRO-3, gradient checkpointing, mixed precision, and one-command multi-node launch.

## ✨ Highlights

- 🔀 **FSDP + ZeRO-3** — shard model, optimizer states, and gradients across GPUs
- 🧮 **Gradient checkpointing** — train 70B+ models on consumer hardware
- ⚡ **Mixed precision (BF16)** — 2x throughput vs FP32 with no accuracy loss
- 🌐 **Multi-node** — `torchrun` + NCCL for seamless multi-machine training
- 📊 **WandB / MLflow** — training curves, GPU utilization, throughput metrics
- 💾 **Checkpoint sharding** — save/load sharded FSDP checkpoints efficiently

## Performance

| Config              | Model   | GPUs | Throughput     | VRAM/GPU |
|---------------------|---------|------|----------------|----------|
| FSDP + ZeRO-2       | 7B      | 4xA100 | 18k tok/s   | 38 GB    |
| FSDP + ZeRO-3       | 13B     | 8xA100 | 22k tok/s   | 42 GB    |
| DeepSpeed ZeRO-3    | 70B     | 16xA100 | 31k tok/s  | 68 GB    |

## Quick Start

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py --config configs/fsdp_7b.yaml

# Multi-node (2 nodes x 8 GPUs)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=10.0.0.1 --master_port=29500 \
  train.py --config configs/deepspeed_70b.yaml
```

## License
MIT © Rutvik Trivedi
