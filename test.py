import torch
from PIL import Image
from diffusers import LongCatImageEditPipeline

if __name__ == '__main__':
    # 关键修改：替换为你本地模型权重的绝对路径
    # 你终端里显示的路径是 /storage/v-jinpewang/az_workspace/wenjun/LongCat-Image-Edit
    local_model_path = "/storage/v-jinpewang/az_workspace/wenjun/LongCat-Image-Edit"

    device = torch.device('cuda')

    # 核心修改：将在线模型名替换为本地路径
    pipe = LongCatImageEditPipeline.from_pretrained(
        local_model_path,  # 使用本地路径
        torch_dtype=torch.bfloat16,
        # 可选：如果加载时遇到安全相关警告，添加下面这行
        # safety_checker=None
    )

    # pipe.to(device, torch.bfloat16)  # 高显存设备取消注释（推理更快）
    pipe.enable_model_cpu_offload()  # 卸载到CPU节省显存（需约18GB）；速度慢但避免显存溢出

    img = Image.open('assets/test.png').convert('RGB')
    prompt = '将猫变成狗'
    image = pipe(
        img,
        prompt,
        negative_prompt='',
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cpu").manual_seed(43)
    ).images[0]

    image.save('./edit_example.png')