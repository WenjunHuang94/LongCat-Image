import os
import sys
import time
import warnings
import argparse
import yaml
import torch
import math
import logging
import datetime  # <--- 【新增】用于生成带时间的 run_name
import wandb  # <--- 【新增】导入 wandb
import transformers
import diffusers
from pathlib import Path
from transformers import Qwen2Model, Qwen2TokenizerFast
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

from train_dataset import build_dataloader
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.utils import LogBuffer
from longcat_image.utils import pack_latents, unpack_latents, calculate_shift, prepare_pos_ids

warnings.filterwarnings("ignore")  # ignore warning

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

logger = get_logger(__name__)


def train(global_step=0):
    # Train!
    total_batch_size = args.train_batch_size * \
                       accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    last_tic = time.time()

    # ==========================================================
    # 【新增】EMA Loss 和 Best Loss 的状态变量
    # ==========================================================
    ema_loss = None
    ema_decay = 0.99
    best_loss = float('inf')

    # Now you train the model
    for epoch in range(first_epoch, args.num_train_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0

        for step, batch in enumerate(train_dataloader):
            image = batch['images']
            ref_image = batch['ref_images']

            data_time_all += time.time() - data_time_start

            with torch.no_grad():
                latents = vae.encode(image.to(weight_dtype).to(accelerator.device)).latent_dist.sample()
                latents = latents.to(dtype=(weight_dtype))
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                ref_latents = vae.encode(ref_image.to(weight_dtype).to(accelerator.device)).latent_dist.sample()
                ref_latents = ref_latents.to(dtype=(weight_dtype))
                ref_latents = (ref_latents - vae.config.shift_factor) * vae.config.scaling_factor

            text_input_ids = batch['input_ids'].to(accelerator.device)
            text_attention_mask = batch['attention_mask'].to(accelerator.device)
            pixel_values = batch['pixel_values'].to(accelerator.device)
            image_grid_thw = batch['image_grid_thw'].to(accelerator.device)

            with torch.no_grad():
                text_output = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True
                )
                prompt_embeds = text_output.hidden_states[-1].clone().detach()

            prompt_embeds = prompt_embeds.to(weight_dtype)
            prompt_embeds = prompt_embeds[:,
                            args.prompt_template_encode_start_idx: -args.prompt_template_encode_end_idx, :]

            grad_norm = None
            with accelerator.accumulate(transformer):
                optimizer.zero_grad()
                sigmas = torch.sigmoid(torch.randn((latents.shape[0],), device=accelerator.device, dtype=latents.dtype))

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
                img_ids = prepare_pos_ids(modality_id=1,
                                          type='image',
                                          start=(prompt_embeds.shape[1], prompt_embeds.shape[1]),
                                          height=latents.shape[2] // 2,
                                          width=latents.shape[3] // 2).to(accelerator.device, dtype=torch.float64)
                img_ids_ref = prepare_pos_ids(modality_id=2,
                                              type='image',
                                              start=(prompt_embeds.shape[1], prompt_embeds.shape[1]),
                                              height=ref_latents.shape[2] // 2,
                                              width=ref_latents.shape[3] // 2).to(accelerator.device,
                                                                                  dtype=torch.float64)

                timesteps = (
                        torch.tensor(timesteps)
                        .expand(noisy_latents.shape[0])
                        .to(device=accelerator.device)
                        / 1000
                )
                text_ids = prepare_pos_ids(modality_id=0,
                                           type='text',
                                           start=(0, 0),
                                           num_token=prompt_embeds.shape[1]).to(accelerator.device, torch.float64)

                img_ids = torch.cat([img_ids, img_ids_ref], dim=0)
                latent_model_input = torch.cat([packed_noisy_latents, packed_ref_latents], dim=1)
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                    model_pred = transformer(latent_model_input, prompt_embeds, timesteps,
                                             img_ids, text_ids, guidance, return_dict=False)[0]
                    model_pred = model_pred[:, :packed_noisy_latents.size(1)]

                model_pred = unpack_latents(
                    model_pred,
                    height=latents.shape[2] * 8,
                    width=latents.shape[3] * 8,
                    vae_scale_factor=16,
                )

                target = noise - latents
                loss = torch.mean(
                    ((model_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    1,
                ).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = transformer.get_global_grad_norm()

                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()

                if accelerator.sync_gradients and args.use_ema:
                    model_ema.step(transformer.parameters())

            lr = lr_scheduler.get_last_lr()[0]

            if accelerator.sync_gradients:
                bsz, ic, ih, iw = image.shape

                # ==========================================================
                # 【新增】Wandb 核心记录逻辑 (仅在主进程中执行，防止多卡冲突)
                # ==========================================================
                current_true_loss = accelerator.gather(loss).mean().item()

                if accelerator.is_main_process:
                    if ema_loss is None:
                        ema_loss = current_true_loss
                    else:
                        ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_true_loss

                    if current_true_loss < best_loss:
                        best_loss = current_true_loss

                    wandb.log({
                        "global_step": global_step,
                        "train_loss": current_true_loss,
                        "train_loss_ema": ema_loss,
                        "best_loss": best_loss,
                        "lr": lr,
                        "epoch": epoch
                    })
                # ==========================================================

                logs = {"loss": current_true_loss, 'aspect_ratio': (ih * 1.0 / iw)}
                if grad_norm is not None:
                    logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())

                log_buffer.update(logs)
                if (step + 1) % args.log_interval == 0 or (step + 1) == 1:
                    t = (time.time() - last_tic) / args.log_interval
                    t_d = data_time_all / args.log_interval

                    log_buffer.average()
                    info = f"Step={step + 1}, Epoch={epoch}, global_step={global_step}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:(ch:{latents.shape[1]},h:{latents.shape[2]},w:{latents.shape[3]}), "
                    info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                    logger.info(info)
                    last_tic = time.time()
                    log_buffer.clear()
                    data_time_all = 0

                # 依然保留 accelerate 自身的日志记录以防不时之需
                logs.update(lr=lr)
                accelerator.log(logs, step=global_step)
                global_step += 1
                data_time_start = time.time()

                if global_step != 0 and global_step % args.save_model_steps == 0:
                    accelerator.wait_for_everyone()
                    os.umask(0o000)
                    cur_lora_ckpt_save_dir = f"{args.work_dir}/checkpoints-{global_step}"
                    os.makedirs(cur_lora_ckpt_save_dir, exist_ok=True)
                    if accelerator.is_main_process:
                        if hasattr(transformer, 'module'):
                            transformer.module.save_pretrained(cur_lora_ckpt_save_dir)
                        else:
                            transformer.save_pretrained(cur_lora_ckpt_save_dir)
                        logger.info(f'Saved lora checkpoint of epoch {epoch} to {cur_lora_ckpt_save_dir}.')
                    accelerator.save_state(cur_lora_ckpt_save_dir)
                    accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break

    # 【新增】训练结束关闭 wandb
    if accelerator.is_main_process:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config", type=str, default='', help="config")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    args = parse_args()

    if args.config != '' and os.path.exists(args.config):
        config = yaml.safe_load(open(args.config, 'r'))
    else:
        config = yaml.safe_load(open(f'{cur_dir}/train_config.yaml', 'r'))

    args_dict = vars(args)
    args_dict.update(config)
    args = argparse.Namespace(**args_dict)

    os.umask(0o000)
    os.makedirs(args.work_dir, exist_ok=True)

    log_dir = args.work_dir + f'/logs'
    os.makedirs(log_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir, logging_dir=log_dir)

    with open(f'{log_dir}/train.yaml', 'w') as f:
        yaml.dump(args_dict, f)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info(f'using weight_dtype {weight_dtype}!!!')

    if args.diffusion_pretrain_weight:
        transformer = LongCatImageTransformer2DModel.from_pretrained(args.diffusion_pretrain_weight,
                                                                     ignore_mismatched_sizes=False)
        logger.info(f'successful load model weight {args.diffusion_pretrain_weight}!!!')
    else:
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "transformer"), ignore_mismatched_sizes=False)
        logger.info(f'successful load model weight {args.pretrained_model_name_or_path + "/transformer"}!!!')

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
        use_rslora=False
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    total_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f">>>>>> total_trainable_params: {total_trainable_params}")

    if args.use_ema:
        model_ema = EMAModel(transformer.parameters(), decay=args.ema_rate)
    else:
        model_ema = None

    vae_dtype = torch.float32
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype).cuda().eval()

    text_encoder = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype,
        trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=weight_dtype, trust_remote_code=True)
    text_processor = AutoProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=weight_dtype, trust_remote_code=True)
    logger.info("all models loaded successfully")

    # build models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    latent_size = int(args.resolution) // 8
    mu = calculate_shift(
        (latent_size // 2) ** 2,
        noise_scheduler.config.base_image_seq_len,
        noise_scheduler.config.max_image_seq_len,
        noise_scheduler.config.base_shift,
        noise_scheduler.config.max_shift,
    )


    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "transformer"))
                if len(weights) != 0:
                    weights.pop()


    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            load_model = LongCatImageTransformer2DModel.from_pretrained(
                input_dir, subfolder="transformer")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = transformer.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    transformer.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        model_ema.to(accelerator.device, dtype=weight_dtype)

    train_dataloader = build_dataloader(args, args.data_txt_root, tokenizer, text_processor, args.resolution, )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.work_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # 兼容绝对路径和相对路径恢复
            full_ckpt_path = os.path.join(os.path.dirname(args.resume_from_checkpoint),
                                          path) if args.resume_from_checkpoint != "latest" else os.path.join(
                args.work_dir, path)
            logger.info(f"Resuming from checkpoint {full_ckpt_path}")
            accelerator.load_state(full_ckpt_path)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    # ==========================================================
    # 【新增】原生 Wandb 初始化代码 (仅在主进程中执行)
    # ==========================================================
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if accelerator.is_main_process:
        current_time = datetime.datetime.now().strftime("%m%d-%H%M")
        # 生成带配置和时间的实验名称
        run_name = f"LongCat-Stage2-r{args.lora_rank}-{current_time}"

        wandb.init(
            project="LongCat-Image-Edit-LoRA-A100-0225",  # 你的项目名称
            name=run_name,
            config=vars(args)  # 把参数上传
        )

        tracker_config = dict(vars(args))
        try:
            accelerator.init_trackers('sft', tracker_config)
        except Exception as e:
            logger.warning(f'get error in save config, {e}')
            accelerator.init_trackers(f"sft_{timestamp}")
    # ==========================================================

    transformer, optimizer, _, _ = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)

    log_buffer = LogBuffer()
    train(global_step=global_step)