#!/bin/bash

# 1. 屏蔽一些烦人的警告
export TOKENIZERS_PARALLELISM=False
export NCCL_DEBUG=WARN  # 改为 WARN，避免刷屏

# 2. 【关键】让脚本看到你所有的 8 张显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 3. 获取路径
script_dir=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd -P)
echo "script_dir: ${script_dir}"

# 4. 直接用 python 运行！不需要 accelerate launch
python ${script_dir}/train_edit_lora_multigpu.py \
    --config ${script_dir}/train_config_multigpuV2.yaml \
    --num_gpus 8