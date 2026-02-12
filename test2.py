import torch
import os
from pathlib import Path
from PIL import Image
from diffusers import LongCatImageEditPipeline


def process_image(pipe, img_path, prompt, save_dir):
    """
    处理单张图片的编辑任务，生成的文件名与原文件一一对应
    Args:
        pipe: LongCatImageEditPipeline 实例
        img_path: 输入图片路径
        prompt: 编辑提示词
        save_dir: 生成图片的保存目录
    """
    try:
        # 打开并处理图片
        img = Image.open(img_path).convert('RGB')

        # 执行图片编辑
        image = pipe(
            img,
            prompt,
            negative_prompt='',
            guidance_scale=4.5,
            num_inference_steps=50,
            num_images_per_prompt=1,
            generator=torch.Generator("cpu").manual_seed(43)
        ).images[0]

        # 获取原始文件名（例如：test1.png）
        original_filename = os.path.basename(img_path)
        # 拼接最终保存路径（指定目录 + 原始文件名）
        save_path = os.path.join(save_dir, original_filename)

        # 保存生成的图片
        image.save(save_path)
        print(f"✅ 成功生成: {save_path} (对应原文件: {img_path})")

    except Exception as e:
        print(f"❌ 处理失败 {img_path}: {str(e)}")


if __name__ == '__main__':
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 初始化管道
    pipe = LongCatImageEditPipeline.from_pretrained(
        "meituan-longcat/LongCat-Image-Edit",
        torch_dtype=torch.bfloat16
    )
    # 显存优化（根据你的显卡显存选择）
    # 高显存设备（24G+）可以取消下面注释，注释掉 enable_model_cpu_offload
    # pipe.to(device, torch.bfloat16)
    pipe.enable_model_cpu_offload()  # 低显存设备（18G左右）必选

    # 3. 定义指定的保存目录（可自行修改）
    SAVE_DIRECTORY = "./longcat_generated_images"

    # 4. 自动创建保存目录（如果不存在）
    Path(SAVE_DIRECTORY).mkdir(parents=True, exist_ok=True)  # 更简洁的创建方式
    print(f"📁 图片将保存到: {SAVE_DIRECTORY} (目录不存在已自动创建)")

    # 5. 定义10组图片路径和对应的prompt
    # 请根据你的实际文件路径和prompt修改这里！
    image_prompt_pairs = [
        ("0206-images/0.png", "Generate a realistic image based on the text description in the image"),
        ("0206-images/1.png", "Modify the image at the annotated location according to the text instruction"),
        ("0206-images/2.png", "Modify the image at the annotated location according to the text instruction"),
        ("0206-images/3.png", "Modify the image at the annotated location according to the text instruction"),
        ("0206-images/4.png", "Follow the visual pointer and text to edit the image"),
        ("0206-images/5.png", "Follow the visual pointer and text to edit the image"),
        ("0206-images/6.png", "Follow the visual pointer and text to edit the image"),
        ("0206-images/7.png", "Edit the image according to the text instruction in the image"),
        ("0206-images/8.png", "Edit the image according to the text instruction in the image"),
        ("0206-images/9.png", "Edit the image according to the text instruction in the image"),
    ]

    # 6. 批量处理所有图片
    for img_path, prompt in image_prompt_pairs:
        # 检查输入图片是否存在
        if not os.path.exists(img_path):
            print(f"⚠️ 图片不存在: {img_path}，跳过")
            continue

        # 处理单张图片（生成的文件名与原文件一致）
        process_image(pipe, img_path, prompt, SAVE_DIRECTORY)

    print("\n🎉 批量处理完成！所有图片已保存至:", SAVE_DIRECTORY)
    print("📌 生成的文件名与原始输入文件名完全一一对应")