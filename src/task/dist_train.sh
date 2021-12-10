#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3,4 python -m torch.distributed.launch --nproc_per_node=3 --master_port=16001 endo_detect_efficient.py
