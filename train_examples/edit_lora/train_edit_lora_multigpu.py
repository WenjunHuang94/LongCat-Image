import os
import sys
import time
import warnings
import math
import logging
from pathlib import Path
from typing import Dict

import argparse
import yaml
import torch
from accelerate import dispatch_model
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel, AutoProcessor

from train_dataset import build_dataloader
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.utils import LogBuffer
from longcat_image.utils import pack_latents, unpack_latents, calculate_shift, prepare_pos_ids


warnings.filterwarnings("ignore")

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

logger = logging.getLogger(__name__)


class MultiGPUTransformerWrapper:
    """
    将 LongCatImageTransformer2DModel 按 block 维度做简单的 model parallel。
    - 非 block 子模块放在 cuda:0
    - transformer_blocks 和 single_transformer_blocks 均匀切到多张卡上
    """

    def __init__(self, transformer: LongCatImageTransformer2DModel, num_gpus: int | None = None):
        self.transformer = transformer
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise RuntimeError("No CUDA devices available for MultiGPUTransformerWrapper.")

        if num_gpus is None or num_gpus <= 0 or num_gpus > available_gpus:
            self.num_gpus = available_gpus
        else:
            self.num_gpus = num_gpus

        self.total_blocks = len(transformer.transformer_blocks)
        self.total_single_blocks = len(transformer.single_transformer_blocks)

        self.split_points = [i * (self.total_blocks // self.num_gpus) for i in range(1, self.num_gpus)]
        self.single_split_points = [
            i * (self.total_single_blocks // self.num_gpus) for i in range(1, self.num_gpus)
        ]

    @property
    def device_map(self) -> Dict[str, str]:
        device_map: Dict[str, str] = {}

        # 1) 非 block 子模块 -> cuda:0
        for name, _ in self.transformer.named_children():
            if name not in ["transformer_blocks", "single_transformer_blocks"]:
                device_map[name] = "cuda:0"

        # 2) transformer_blocks
        res = 0
        for item, splt in enumerate(self.split_points):
            temp = {f"transformer_blocks.{i}": f"cuda:{min(item + 1, self.num_gpus - 1)}" for i in range(res, splt)}
            res = splt
            device_map.update(temp)
        temp = {f"transformer_blocks.{i}": f"cuda:{self.num_gpus - 1}" for i in range(res, self.total_blocks)}
        device_map.update(temp)

        # 3) single_transformer_blocks
        res = 0
        for item, splt in enumerate(self.single_split_points):
            temp = {
                f"single_transformer_blocks.{i}": f"cuda:{min(item + 1, self.num_gpus - 1)}"
                for i in range(res, splt)
            }
            res = splt
            device_map.update(temp)
        temp = {
            f"single_transformer_blocks.{i}": f"cuda:{self.num_gpus - 1}"
            for i in range(res, self.total_single_blocks)
        }
        device_map.update(temp)

        return device_map

    def auto_split(self):
        try:
            model = dispatch_model(self.transformer, device_map=self.device_map)
            print(f"[MultiGPU] Successfully applied device_map across {self.num_gpus} GPUs.")
        except Exception as e:
            print(f"[MultiGPU] Error with accelerate dispatch: {e}")
            model = self.transformer
        return model


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU LoRA training for LongCat-Image-Edit.")
    parser.add_argument("--config", type=str, default="", help="Path to train_config.yaml")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use for model-parallel transformer. 0 = use all available.",
    )
    args = parser.parse_args()
    return args


def main():
    # basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU training.")

    args_cli = parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if args_cli.config and os.path.exists(args_cli.config):
        config = yaml.safe_load(open(args_cli.config, "r"))
    else:
        config = yaml.safe_load(open(f"{cur_dir}/train_config.yaml", "r"))

    # 将 yaml 中的配置展平为 Namespace
    args_dict = dict(config)
    # 额外添加 CLI 参数
    args_dict.update(vars(args_cli))
    args = argparse.Namespace(**args_dict)

    os.umask(0o000)
    os.makedirs(args.work_dir, exist_ok=True)

    # 设置随机种子
    if getattr(args, "seed", None) is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # 选择精度
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    logger.info(f"Using weight_dtype = {weight_dtype}")

    # 1. 加载基础 transformer（在 CPU 上）
    if args.diffusion_pretrain_weight:
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            args.diffusion_pretrain_weight, ignore_mismatched_sizes=False
        )
        logger.info(f"Loaded transformer from diffusion_pretrain_weight = {args.diffusion_pretrain_weight}")
    else:
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "transformer"), ignore_mismatched_sizes=False
        )
        logger.info(f"Loaded transformer from {args.pretrained_model_name_or_path + '/transformer'}")

    # 2. 多卡切分 transformer
    wrapper = MultiGPUTransformerWrapper(transformer, num_gpus=args_cli.num_gpus)
    transformer_mp = wrapper.auto_split()

    # 3. 注入 LoRA
    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
        use_dora=False,
        use_rslora=False,
    )
    transformer_mp = get_peft_model(transformer_mp, lora_config)
    transformer_mp.print_trainable_parameters()

    total_trainable_params = sum(p.numel() for p in transformer_mp.parameters() if p.requires_grad)
    logger.info(f">>>>>> total_trainable_params: {total_trainable_params}")

    if args.gradient_checkpointing:
        transformer_mp.enable_gradient_checkpointing()

    # 4. 加载 VAE & 文本编码器（放到 cuda:0）
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
    ).to("cuda:0").eval()

    text_encoder = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    ).to("cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", trust_remote_code=True
    )
    text_processor = AutoProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", trust_remote_code=True
    )
    logger.info("All models loaded successfully (multi-GPU transformer, VAE, text encoder).")

    # 5. scheduler & mu
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    latent_size = int(args.resolution) // 8
    mu = calculate_shift(
        (latent_size // 2) ** 2,
        noise_scheduler.config.base_image_seq_len,
        noise_scheduler.config.max_image_seq_len,
        noise_scheduler.config.base_shift,
        noise_scheduler.config.max_shift,
    )

    # 6. dataloader
    train_dataloader = build_dataloader(args, args.data_txt_root, tokenizer, text_processor, args.resolution)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 7. optimizer & scheduler
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install bitsandbytes: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [p for p in transformer_mp.parameters() if p.requires_grad]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 8. 训练循环
    global_step = 0
    log_buffer = LogBuffer()
    last_tic = time.time()

    logger.info("***** Running multi-GPU LoRA training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Train batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    for epoch in range(num_train_epochs):
        data_time_start = time.time()
        data_time_all = 0.0

        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            image = batch["images"]  # (B, C, H, W)
            ref_image = batch["ref_images"]

            data_time_all += time.time() - data_time_start

            with torch.no_grad():
                latents = vae.encode(image.to(weight_dtype).to("cuda:0")).latent_dist.sample()
                latents = latents.to(dtype=weight_dtype)
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                ref_latents = vae.encode(ref_image.to(weight_dtype).to("cuda:0")).latent_dist.sample()
                ref_latents = ref_latents.to(dtype=weight_dtype)
                ref_latents = (ref_latents - vae.config.shift_factor) * vae.config.scaling_factor

            text_input_ids = batch["input_ids"].to("cuda:0")
            text_attention_mask = batch["attention_mask"].to("cuda:0")
            pixel_values = batch["pixel_values"].to("cuda:0")
            image_grid_thw = batch["image_grid_thw"].to("cuda:0")

            with torch.no_grad():
                text_output = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                )
                prompt_embeds = text_output.hidden_states[-1].clone().detach()

            prompt_embeds = prompt_embeds.to(weight_dtype)
            prompt_embeds = prompt_embeds[
                :, args.prompt_template_encode_start_idx : -args.prompt_template_encode_end_idx, :
            ]

            optimizer.zero_grad()

            # loRA 训练核心
            sigmas = torch.sigmoid(torch.randn((latents.shape[0],), device="cuda:0", dtype=latents.dtype))
            if args.use_dynamic_shifting:
                sigmas = noise_scheduler.time_shift(mu, 1.0, sigmas)

            timesteps = sigmas * 1000.0
            sigmas = sigmas.view(-1, 1, 1, 1)

            noise = torch.randn_like(latents)

            noisy_latents = (1 - sigmas) * latents + sigmas * noise
            noisy_latents = noisy_latents.to(weight_dtype)

            packed_noisy_latents = pack_latents(
                noisy_latents,
                batch_size=latents.shape[0],
                num_channels_latents=latents.shape[1],
                height=latents.shape[2],
                width=latents.shape[3],
            )

            packed_ref_latents = pack_latents(
                ref_latents,
                batch_size=ref_latents.shape[0],
                num_channels_latents=ref_latents.shape[1],
                height=ref_latents.shape[2],
                width=ref_latents.shape[3],
            )

            guidance = None
            img_ids = prepare_pos_ids(
                modality_id=1,
                type="image",
                start=(prompt_embeds.shape[1], prompt_embeds.shape[1]),
                height=latents.shape[2] // 2,
                width=latents.shape[3] // 2,
            ).to("cuda:0", dtype=torch.float64)
            img_ids_ref = prepare_pos_ids(
                modality_id=2,
                type="image",
                start=(prompt_embeds.shape[1], prompt_embeds.shape[1]),
                height=ref_latents.shape[2] // 2,
                width=ref_latents.shape[3] // 2,
            ).to("cuda:0", dtype=torch.float64)

            timesteps = torch.tensor(timesteps).expand(noisy_latents.shape[0]).to(device="cuda:0") / 1000

            text_ids = prepare_pos_ids(
                modality_id=0,
                type="text",
                start=(0, 0),
                num_token=prompt_embeds.shape[1],
            ).to("cuda:0", dtype=torch.float64)

            img_ids = torch.cat([img_ids, img_ids_ref], dim=0)
            latent_model_input = torch.cat([packed_noisy_latents, packed_ref_latents], dim=1).to(weight_dtype)

            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                model_pred = transformer_mp(
                    latent_model_input,
                    prompt_embeds,
                    timesteps,
                    img_ids,
                    text_ids,
                    guidance,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_latents.size(1)]

            model_pred = unpack_latents(
                model_pred,
                height=latents.shape[2] * 8,
                width=latents.shape[3] * 8,
                vae_scale_factor=16,
            )

            target = noise - latents
            loss = torch.mean(
                ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            ).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
            optimizer.step()
            lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]

            bsz, ic, ih, iw = image.shape
            logs = {"loss": loss.detach().item(), "aspect_ratio": (ih * 1.0 / iw)}
            logs["lr"] = lr

            log_buffer.update(logs)
            if (step + 1) % args.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / args.log_interval
                t_d = data_time_all / args.log_interval

                log_buffer.average()
                info = (
                    f"Step={step+1}, Epoch={epoch}, global_step={global_step}, "
                    f"time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "
                    f"s:(ch:{latents.shape[1]},h:{latents.shape[2]},w:{latents.shape[3]}), "
                )
                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0

            global_step += 1
            data_time_start = time.time()

            if global_step != 0 and global_step % args.save_model_steps == 0:
                os.umask(0o000)
                cur_lora_ckpt_save_dir = f"{args.work_dir}/checkpoints-{global_step}"
                os.makedirs(cur_lora_ckpt_save_dir, exist_ok=True)
                # transformer_mp 是 PeftModel，多卡分片也可以直接 save_pretrained
                transformer_mp.save_pretrained(cur_lora_ckpt_save_dir)
                logger.info(f"Saved multi-GPU LoRA checkpoint to {cur_lora_ckpt_save_dir}")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()

