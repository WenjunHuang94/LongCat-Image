import os
import argparse

import torch
from PIL import Image
from accelerate import dispatch_model
from peft import PeftModel

from diffusers import LongCatImageEditPipeline, LongCatImageTransformer2DModel


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU inference for LongCat-Image-Edit with optional LoRA.")

    # model & lora
    parser.add_argument(
        "--model_path",
        type=str,
        default="/storage/v-jinpewang/az_workspace/wenjun/LongCat-Image-Edit",
        help="Base LongCat-Image-Edit model path (HF repo or local directory).",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/storage/v-jinpewang/az_workspace/wenjun/LongCat-Image/output/edit_lora_model/checkpoints-100",
        help="LoRA checkpoint directory (e.g. output/edit_lora_model/checkpoints-500). Leave empty to disable LoRA.",
    )

    # input / output
    parser.add_argument(
        "--input_image",
        type=str,
        default="assets/test.png",
        help="Input image path for editing.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="将猫变成狗",
        help="Edit instruction prompt.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt.",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="edit_lora_multigpu.png",
        help="Output image path.",
    )

    # sampling
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # multi-gpu
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use for transformer model parallel. 0 = use all available.",
    )

    return parser.parse_args()


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

        # 均匀切分主 transformer_blocks
        self.split_points = [i * (self.total_blocks // self.num_gpus) for i in range(1, self.num_gpus)]
        # 对 single_transformer_blocks 也做类似切分
        self.single_split_points = [
            i * (self.total_single_blocks // self.num_gpus) for i in range(1, self.num_gpus)
        ]

    @property
    def device_map(self):
        device_map: dict[str, str] = {}

        # 1) 非 block 子模块 -> cuda:0
        for name, _ in self.transformer.named_children():
            if name not in ["transformer_blocks", "single_transformer_blocks"]:
                device_map[name] = "cuda:0"

        # 2) transformer_blocks 均匀分到各 GPU
        res = 0
        for item, splt in enumerate(self.split_points):
            temp = {f"transformer_blocks.{i}": f"cuda:{min(item + 1, self.num_gpus - 1)}" for i in range(res, splt)}
            res = splt
            device_map.update(temp)

        temp = {
            f"transformer_blocks.{i}": f"cuda:{self.num_gpus - 1}"
            for i in range(res, self.total_blocks)
        }
        device_map.update(temp)

        # 3) single_transformer_blocks 均匀分到各 GPU（策略相同）
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


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU inference.")

    dtype = torch.bfloat16

    # 1. 先加载基础 transformer 到 CPU（后面再做 model parallel）
    print(f"[Init] Loading base transformer from {args.model_path} ...")
    base_transformer = LongCatImageTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # 2. 多卡切分 transformer
    wrapper = MultiGPUTransformerWrapper(base_transformer, num_gpus=args.num_gpus)
    mp_transformer = wrapper.auto_split()

    # 3. （可选）在多卡 transformer 上加载 LoRA
    if args.lora_path:
        if os.path.exists(args.lora_path):
            print(f"[LoRA] Loading LoRA from {args.lora_path} using PEFT...")

            def _unwrap(m):
                return m._orig_mod if hasattr(m, "_orig_mod") else m

            base_for_lora = _unwrap(mp_transformer)
            mp_transformer = PeftModel.from_pretrained(base_for_lora, args.lora_path, low_cpu_mem_usage=False)
            mp_transformer.eval()
            print("[LoRA] LoRA weights loaded.")
        else:
            print(f"[LoRA] WARNING: LoRA path '{args.lora_path}' does not exist, skip loading.")

    # 4. 构建 LongCatImageEditPipeline，并注入多卡 transformer
    print("[Init] Creating LongCatImageEditPipeline ...")
    pipe = LongCatImageEditPipeline.from_pretrained(
        args.model_path,
        transformer=mp_transformer,
        torch_dtype=dtype,
    )

    # 将 VAE / text encoder 至少放到 cuda:0，避免频繁 CPU-GPU 拷贝
    pipe.vae.to("cuda:0", dtype=dtype)
    pipe.text_encoder.to("cuda:0", dtype=dtype)

    # 关闭 CPU offload（我们已经做了 model parallel）
    # 如需进一步省显存，可以改为 enable_sequential_cpu_offload，但会变慢。
    # pipe.enable_model_cpu_offload()

    # 5. 准备输入
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = Image.open(args.input_image).convert("RGB")

    pipe.set_progress_bar_config(disable=None)

    print("[Infer] Running inference ...")
    with torch.inference_mode():
        output = pipe(
            image,
            args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
        )
        out_img = output.images[0]
        out_img.save(args.output_image)

    print(f"[Done] Image successfully saved to {args.output_image}")


if __name__ == "__main__":
    main()

