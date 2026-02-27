import os
import argparse
import torch
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image

# 引入 LongCat 和 PEFT 依赖
from diffusers import LongCatImageEditPipeline, LongCatImageTransformer2DModel
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="GEdit-Bench Inference with LongCat Image Edit")

    # === 模型路径参数 ===
    parser.add_argument("--pretrained_model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_weight", type=str, default=None, help="LoRA weight path")

    # === 任务与输出参数 ===
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local GEdit-Bench dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for directory structure")
    parser.add_argument("--output_root", type=str, default="results", help="Root directory for saving results")
    parser.add_argument("--num_images", type=int, default=None, help="Debug: Number of images to process")

    # === 推理超参数 ===
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    # ==========================================
    # 1. 初始化 LongCat 模型与 LoRA
    # ==========================================
    print(f"Loading base transformer from {args.pretrained_model}...")
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder='transformer',
        torch_dtype=dtype,
        use_safetensors=True
    ).to(device)

    if args.lora_weight and os.path.exists(args.lora_weight):
        print(f"Loading LoRA from {args.lora_weight} using PEFT...")
        transformer = PeftModel.from_pretrained(transformer, args.lora_weight)
        transformer = transformer.merge_and_unload()
        transformer.to(device, dtype=dtype)
        print("LoRA weights merged successfully!")
    elif args.lora_weight:
        raise FileNotFoundError(f"❌ 致命错误: 找不到 LoRA 权重！请检查路径: {args.lora_weight}")

    print("Creating pipeline...")
    pipe = LongCatImageEditPipeline.from_pretrained(
        args.pretrained_model,
        transformer=transformer,
        torch_dtype=dtype
    )

    # 根据显存情况选择是否 offload。如果显存够大（如 40GB+），建议注释掉这行并使用 pipe.to(device) 以加速推理
    pipe.enable_model_cpu_offload()

    # 注意：如果启用了 cpu_offload，generator 通常仍应放在 CUDA 上以保证生成质量与速度
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ==========================================
    # 2. 加载 GEdit-Bench 数据集
    # ==========================================
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    data_split = dataset

    if args.num_images is not None and args.num_images > 0:
        limit = min(args.num_images, len(data_split))
        print(f"🛠️ Debug Mode: 只运行前 {limit} 张图片")
        data_split = data_split.shuffle(seed=args.seed).select(range(limit))

    print(f"Start Inference. Total images: {len(data_split)}")

    # ==========================================
    # 3. 循环推理与按规范保存
    # ==========================================
    for item in tqdm(data_split, desc="Inferencing"):
        image_input = item['input_image']
        prompt = item['instruction']
        task_type = item['task_type']
        lang = item['instruction_language']
        file_key = item['key']

        # 构建符合 GEdit-Bench 标准的路径
        lang_folder = 'cn' if lang in ['zh', 'chinese', 'cn'] else 'en'
        save_dir = os.path.join(args.output_root, args.model_name, "fullset", task_type, lang_folder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_key}.png")

        # 避免重复生成
        if os.path.exists(save_path):
            continue

        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        # LongCat 推理
        with torch.inference_mode():
            output = pipe(
                image=image_input,
                prompt=prompt,
                negative_prompt=args.neg_prompt,
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.infer_steps,
                num_images_per_prompt=1,
                generator=generator
            )
            output.images[0].save(save_path)

    print(f"\n✅ All tasks completed. Results saved in: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()