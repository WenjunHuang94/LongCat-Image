import os
import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 引入 LongCat 和 PEFT 依赖
from diffusers import LongCatImageEditPipeline, LongCatImageTransformer2DModel
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="CEdit-Bench Inference with LongCat Image Edit")

    # === 模型路径参数 ===
    parser.add_argument("--pretrained_model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_weight", type=str, default=None, help="LoRA weight path")

    # === 任务与输出参数 ===
    parser.add_argument("--dataset_name", type=str, default="meituan-longcat/CEdit-Bench",
                        help="HuggingFace dataset name or local path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for directory structure")
    parser.add_argument("--output_root", type=str, default="edit_images", help="Root directory for saving results")
    parser.add_argument("--lang", type=str, default="both", choices=["en", "cn", "both"],
                        help="Which language instruction to infer")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Debug: Number of images to process per task category")
    parser.add_argument("--task_type", type=str, default="all",
                        help="指定只跑哪种类型的任务 (如 'Background Change')，默认 all")

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

    # 启用 offload 节省显存，显存富裕时可注释以加速
    pipe.enable_model_cpu_offload()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ==========================================
    # 2. 加载与解析 CEdit-Bench 数据集
    # ==========================================
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")

    # 构建待推理的任务列表（处理双语）
    inference_tasks = []

    # 用于 Debug 模式的类别计数
    from collections import defaultdict
    task_counts = defaultdict(int)

    # 如果设置了抽样，为了保证随机性，先打乱数据集
    if args.num_images is not None and args.num_images > 0:
        dataset = dataset.shuffle(seed=args.seed)

    for item in dataset:
        category = item['task_type']

        # === 核心修改：如果指定了类别，且当前类别不匹配，则跳过 ===
        if args.task_type != "all" and category != args.task_type:
            continue

        # 类别数量控制
        if args.num_images is not None and args.num_images > 0:
            if task_counts[category] >= args.num_images:
                continue
            task_counts[category] += 1

        # CEdit-Bench 的核心差异：拆分英文和中文的 prompt
        if args.lang in ['en', 'both']:
            inference_tasks.append({
                'input_image': item['input_image'],
                'prompt': item['instruction_en'],
                'task_type': category,
                'lang': 'en',
                'key': item['key']
            })

        if args.lang in ['cn', 'both']:
            inference_tasks.append({
                'input_image': item['input_image'],
                'prompt': item['instruction_cn'],
                'task_type': category,
                'lang': 'cn',
                'key': item['key']
            })

    print(f"✅ 数据准备完成！总共待推理任务数: {len(inference_tasks)}。")
    if args.num_images is not None and args.num_images > 0:
        print(f"📊 抽样任务分布 (基于原图): {dict(task_counts)}")

    # ==========================================
    # 3. 循环推理与按规范保存
    # ==========================================
    for item in tqdm(inference_tasks, desc="Inferencing"):
        image_input = item['input_image']
        prompt = item['prompt']
        task_type = item['task_type']
        lang = item['lang']
        file_key = item['key']

        # 路径规范：output_root / model_name / fullset / task_type / lang
        save_dir = os.path.join(args.output_root, args.model_name, "fullset", task_type, lang)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_key}.png")

        # 避免中断后重复生成
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

    print(f"\n🎉 All tasks completed. Results saved in: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()