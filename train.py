"""Distributed LLM training entry point."""
import os, argparse
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import TrainingConfig
from src.data import get_dataloader
from src.trainer import Trainer


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def wrap_with_fsdp(model, config):
    from functools import partial
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    return FSDP(model, auto_wrap_policy=auto_wrap, cpu_offload=CPUOffload(offload_params=config.cpu_offload),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE, mixed_precision=None, device_id=torch.cuda.current_device())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = TrainingConfig.from_yaml(args.config)
    setup_distributed()
    rank = dist.get_rank()
    if rank == 0: print(f"Training {config.model_name} on {dist.get_world_size()} GPUs")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
    model = wrap_with_fsdp(model, config)
    dataloader = get_dataloader(config, tokenizer, rank)
    trainer = Trainer(model, tokenizer, dataloader, config)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
