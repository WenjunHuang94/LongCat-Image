export TOKENIZERS_PARALLELISM=False
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=12000

# 【新增】指定只使用第 0 号显卡 (如果你想用第1号，就写=1)
export CUDA_VISIBLE_DEVICES=0

script_dir=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd -P)
project_root=$(dirname "$(dirname "$script_dir")")
echo "script_dir" ${script_dir}

deepspeed_config_file=${project_root}/misc/accelerate_config.yaml

accelerate launch  --mixed_precision bf16 --num_processes 1 --config_file ${deepspeed_config_file} \
${script_dir}/train_edit_lora.py \
--config ${script_dir}/train_configv2.yaml