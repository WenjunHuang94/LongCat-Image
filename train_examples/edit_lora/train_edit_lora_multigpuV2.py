import os
import sys
import time
import warnings
import argparse
import yaml
import torch
import math
import logging
import transformers
import diffusers
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from peft import LoraConfig, get_peft_model

# 假设这些是你自己的库
from train_dataset import build_dataloader
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.utils import LogBuffer
from longcat_image.utils import pack_latents, unpack_latents, calculate_shift, prepare_pos_ids

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument("--config", type=str, default='', help="config file path")
    # 移除 local_rank 等参数，accelerate 会自动处理
    args = parser.parse_args()
    return args


def main():
    args_cli = parse_args()

    # 载入配置
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if args_cli.config != '' and os.path.exists(args_cli.config):
        config_path = args_cli.config
    else:
        config_path = f'{cur_dir}/train_config.yaml'

    config = yaml.safe_load(open(config_path, 'r'))
    args_dict = vars(args_cli)
    args_dict.update(config)
    args = argparse.Namespace(**args_dict)

    # 1. 初始化 Accelerator (关键步骤)
    # DeepSpeed 的配置会自动从 accelerate launch 的配置中读取
    accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir, logging_dir=f"{args.work_dir}/logs")

    # 增加超时时间，防止 deepspeed 初始化在大模型时超时
    kwargs = InitProcessGroupKwargs(timeout=torch.tensor(3600))

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    # 2. 准备模型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load Transformer
    if args.diffusion_pretrain_weight:
        transformer = LongCatImageTransformer2DModel.from_pretrained(args.diffusion_pretrain_weight)
    else:
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "transformer"))

    # 开启 Gradient Checkpointing (省显存的关键)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # LoRA Config
    lora_config = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
        ],
        use_dora=False,
        use_rslora=False
    )
    transformer = get_peft_model(transformer, lora_config)

    # 确保 trainable params 是 float32 (为了数值稳定性)
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # 3. 准备其他组件 (VAE/TextEncoder 不需要训练，不放入 prepare)
    # 注意：不要手动 .to(device)，accelerate 会处理，或者在使用时处理
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    text_encoder = AutoModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",
                                             torch_dtype=weight_dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",
                                              trust_remote_code=True)
    text_processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",
                                                   trust_remote_code=True)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path,
                                                                      subfolder="scheduler")

    # 移动冻结的模型到 GPU
    vae.to(accelerator.device).eval()
    text_encoder.to(accelerator.device).eval()

    # 4. Optimizer
    # 如果使用 DeepSpeed，通常不需要 8bit Adam，DeepSpeed 有自己的优化器实现
    # 但如果显存极度紧张，可以使用 bitsandbytes
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 5. Dataloader
    train_dataloader = build_dataloader(args, args.data_txt_root, tokenizer, text_processor, args.resolution)

    # 6. Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # 7. Prepare Everything with Accelerator
    # 注意：DeepSpeed 模式下，model, optimizer, dataloader 必须全部传给 prepare
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # 计算 shift (逻辑保持不变)
    latent_size = int(args.resolution) // 8
    mu = calculate_shift(
        (latent_size // 2) ** 2,
        noise_scheduler.config.base_image_seq_len,
        noise_scheduler.config.max_image_seq_len,
        noise_scheduler.config.base_shift,
        noise_scheduler.config.max_shift,
    )

    # 开始训练循环
    global_step = 0
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total batch size = {total_batch_size}")

    log_buffer = LogBuffer()
    last_tic = time.time()

    for epoch in range(args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Data processing
                image = batch['images'].to(accelerator.device)
                ref_image = batch['ref_images'].to(accelerator.device)

                # VAE Encoding (no grad)
                with torch.no_grad():
                    latents = vae.encode(image.to(weight_dtype)).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                    ref_latents = vae.encode(ref_image.to(weight_dtype)).latent_dist.sample()
                    ref_latents = (ref_latents - vae.config.shift_factor) * vae.config.scaling_factor

                    # Text Encoding
                    text_input_ids = batch['input_ids'].to(accelerator.device)
                    text_attention_mask = batch['attention_mask'].to(accelerator.device)
                    pixel_values = batch['pixel_values'].to(accelerator.device)  # 如果有
                    image_grid_thw = batch['image_grid_thw'].to(accelerator.device)  # 如果有

                    text_output = text_encoder(
                        input_ids=text_input_ids,
                        attention_mask=text_attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True
                    )
                    prompt_embeds = text_output.hidden_states[-1]
                    # 注意：如果使用了 DDP/DeepSpeed，这里切片操作没问题
                    prompt_embeds = prompt_embeds[:,
                                    args.prompt_template_encode_start_idx: -args.prompt_template_encode_end_idx, :]

                # ... (中间生成 noise, sigmas, timesteps 等逻辑保持不变，省略以节省空间) ...
                # 请将原代码中生成 noise, noisy_latents, packed_latents 等逻辑复制到这里
                # 确保所有新建的 tensor 如 sigmas, noise 都使用 device=accelerator.device

                # 示例 noise 生成:
                sigmas = torch.sigmoid(torch.randn((latents.shape[0],), device=accelerator.device, dtype=latents.dtype))
                # ...

                # Forward Pass
                # 注意：transformer 已经被 wrap 成了 DDP/DeepSpeed 模型
                model_pred = transformer(latent_model_input, prompt_embeds, timesteps, img_ids, text_ids, guidance,
                                         return_dict=False)[0]
                model_pred = model_pred[:, :packed_noisy_latents.size(1)]

                # Loss
                # ... unpack logic ...
                target = noise - latents
                loss = torch.mean(((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.sync_gradients:
                global_step += 1
                # logs ...
                if global_step % args.save_model_steps == 0:
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.work_dir, f"checkpoint-{global_step}")
                    # DeepSpeed 保存状态
                    accelerator.save_state(save_path)
                    # 如果只需要 LoRA 权重:
                    if accelerator.is_main_process:
                        unwrap_model = accelerator.unwrap_model(transformer)
                        unwrap_model.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()