import torch
import sys
import os
from PIL import Image

from diffusers import LongCatImageEditPipeline, LongCatImageTransformer2DModel
from peft import PeftModel

if __name__ == '__main__':
    device = torch.device('cuda')

    # 1. 设置模型路径和 LoRA 路径
    model_path = "/home/disk2/hwj/my_hf_cache/hub/models--meituan-longcat--LongCat-Image-Edit/snapshots/7b54ef423aa7854be7861600024be5c56ab7875a"  # 或者本地路径，如 "./weights/LongCat-Image-Edit"
    lora_ckpt_path = "./output/edit_lora_model/checkpoints-500"  # 你的 LoRA 权重路径
    
    # 2. 加载基础 transformer 模型
    print(f"Loading base transformer from {model_path}...")
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        model_path, 
        subfolder='transformer', 
        torch_dtype=torch.bfloat16, 
        use_safetensors=True
    ).to(device)

    # 3. 加载 LoRA 权重
    if lora_ckpt_path and os.path.exists(lora_ckpt_path):
        print(f"Loading LoRA from {lora_ckpt_path} using PEFT...")
        transformer = PeftModel.from_pretrained(transformer, lora_ckpt_path)
        # 合并 LoRA 到基础模型（可选，合并后推理更快，但无法再卸载 LoRA）
        transformer = transformer.merge_and_unload()
        transformer.to(device, dtype=torch.bfloat16)
        print("LoRA weights merged successfully!")
    else:
        print(f"Warning: LoRA checkpoint not found at {lora_ckpt_path}")
    
    # 4. 创建 pipeline，传入加载了 LoRA 的 transformer
    print("Creating pipeline...")
    pipe = LongCatImageEditPipeline.from_pretrained(
        model_path, 
        transformer=transformer, 
        torch_dtype=torch.bfloat16
    )
    # pipe.to(device, torch.bfloat16)  # 高显存设备可以取消注释（推理更快）
    pipe.enable_model_cpu_offload()  # 卸载到 CPU 以节省显存（需要约 18 GB）；较慢但防止 OOM

    # 5. 进行推理
    print("Running inference...")
    img = Image.open('/home/disk2/hwj/LongCat-Image/data_example/origin_images/image_000000.JPEG').convert('RGB')  # 你的输入图片路径
    prompt = 'Generate a realistic image based on the text description in the image'  # 你的编辑指令
    
    image = pipe(
        img,
        prompt,
        negative_prompt='',
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cpu").manual_seed(43)
    ).images[0]

    # 6. 保存结果
    output_path = './edit_lora_example.png'
    image.save(output_path)
    print(f"✅ Inference completed! Result saved to {output_path}")
