import torch
import sys
import os
from PIL import Image

from diffusers import LongCatImageEditPipeline, LongCatImageTransformer2DModel
from peft import PeftModel

if __name__ == '__main__':
    device = torch.device('cuda')

    # 1. 设置模型路径和 LoRA 路径
    model_path = "/home/disk2/hwj/my_hf_cache/hub/models--meituan-longcat--LongCat-Image-Edit/snapshots/7b54ef423aa7854be7861600024be5c56ab7875a"
    lora_ckpt_path = "/home/disk2/hwj/LongCat-Image/output/edit_lora_model-stage2-final-0225/checkpoints-10000"

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
        transformer = transformer.merge_and_unload()
        transformer.to(device, dtype=torch.bfloat16)
        print("✅ LoRA weights merged successfully!")
    else:
        print("⚠️ 未设置 LoRA 路径，将使用基础模型进行推理")
        # raise FileNotFoundError(f"❌ 致命错误: 找不到 LoRA 权重！请检查路径是否正确: {lora_ckpt_path}")

    # 4. 创建 pipeline
    print("Creating pipeline...")
    pipe = LongCatImageEditPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    # ==========================================
    # 5. 定义批量推理任务列表与统一尾缀
    # ==========================================
    UNIFIED_SUFFIX = "-res-03007"  # <--- 【新增】在这里统一定义你的尾缀
    OUTPUT_DIR = "./res-0307-10000"  # <--- 【新增】统一定义输出文件夹

    # 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 现在的 tasks 列表变得更清爽了，只需要填输入图、Prompt 和 CFG
    tasks = [
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/text-to-image-2M-export_4000/input/image_000000.JPEG',
        #     "prompt": 'Generate a realistic image based on the text description in the image',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/text-to-image-2M-export_4000/input/image_000002.JPEG',
        #     "prompt": 'Generate a realistic image based on the text description in the image',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/text-to-image-2M-export_4000/input/image_000003.JPEG',
        #     "prompt": 'Generate a realistic image based on the text description in the image',
        # },
        #
        #
        #
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/000001_00000303_textbox.png',
        #     "prompt": 'Modify the image at the annotated location according to the text instruction',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/000004_00000755_textbox.png',
        #     "prompt": 'Modify the image at the annotated location according to the text instruction',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/000005_00000906_textbox.png',
        #     "prompt": 'Modify the image at the annotated location according to the text instruction',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/00000005.png',
        #     "prompt": 'Follow the visual pointer and text to edit the image',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/00000303.png',
        #     "prompt": 'Follow the visual pointer and text to edit the image',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/task_obj_add_108-1.png',
        #     "prompt": 'Edit the image according to the text instruction in the image',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/task_obj_remove_108.png',
        #     "prompt": 'Follow the visual pointer and text to edit the image',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/task_obj_remove_10773.png',
        #     "prompt": 'Follow the visual pointer and text to edit the image',
        # },
        # {
        #     "img_path": '/home/disk2/hwj/LongCat-Image/hwj_images/task_obj_swap_joint_mask_9812-1.png',
        #     "prompt": 'Follow the visual pointer and text to edit the image',
        # },
        #
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/1.png",
        #     "prompt": "Extract the cat from the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/2.png",
        #     "prompt": "Add another creature next to the main character"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/3.png",
        #     "prompt": "A robot holds the device in its arms across its chest"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/4.png",
        #     "prompt": "Turn the object in red box into a pirate ship"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/5.png",
        #     "prompt": "A wooden table with a pot of flowers, the flowers need to be consistent with the\nspecies and color of the flowers in the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/6.png",
        #     "prompt": "Change the camera angle to a low angle, looking up at the dog"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/7.png",
        #     "prompt": "Let this man wear sunglasses and make a heart gesture with his hands"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/8.png",
        #     "prompt": "Transform the image into a retro-colored illustration style"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/9.png",
        #     "prompt": "Remove the laptop, add a mug, change the wall to yellow, add a girl sitting on the bed, turn on the TV, and change the style to Ghibli."
        # },
        #
        #
        #
        #
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/10.png",
        #     "prompt": "Edit the image according to the text instruction in the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/11.png",
        #     "prompt": "Edit the image according to the text instruction in the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/12.png",
        #     "prompt": "Follow the visual pointer and text to edit the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/13.png",
        #     "prompt": "Follow the visual pointer and text to edit the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/14.png",
        #     "prompt": "Follow the visual pointer and text to edit the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0204-images/15.png",
        #     "prompt": "Follow the visual pointer and text to edit the image"
        # }

        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0301-images/1.png",
        #     "prompt": "Modify the image at the annotated location according to the text instruction"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0301-images/2.png",
        #     "prompt": "Modify the image at the annotated location according to the text instruction"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0301-images/3.png",
        #     "prompt": "Follow the visual pointer and text to edit the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0301-images/4.png",
        #     "prompt": "Follow the visual pointer and text to edit the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0301-images/5.png",
        #     "prompt": "Modify the image at the annotated location according to the text instruction"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0301-images/6.png",
        #     "prompt": "Generate a realistic image based on the text description in the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0301-images/7.png",
        #     "prompt": "Edit the image according to the text instruction in the image"
        # }

        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0303-images/1.png",
        #     "prompt": "Modify the image at the annotated location according to the text instruction"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0303-images/2.png",
        #     "prompt": "Follow the visual pointer and text to edit the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0303-images/3.png",
        #     "prompt": "Follow the visual pointer and text to edit the image"
        # },
        # {
        #     "img_path": "/home/disk2/hwj/LongCat-Image/0303-images/4.png",
        #     "prompt": "Modify the image at the annotated location according to the text instruction"
        # }

        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/1.png",
            "prompt": "Generate a realistic image based on the text description in the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/2.png",
            "prompt": "Generate a realistic image based on the text description in the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/3.png",
            "prompt": "Generate a realistic image based on the text description in the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/4.png",
            "prompt": "Generate a realistic image based on the text description in the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/5.png",
            "prompt": "Generate a realistic image based on the text description in the image"
        },



        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/6.png",
            "prompt": "Edit the image according to the text instruction in the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/7.png",
            "prompt": "Edit the image according to the text instruction in the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/8.png",
            "prompt": "Edit the image according to the text instruction in the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/9.png",
            "prompt": "Edit the image according to the text instruction in the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/10.png",
            "prompt": "Edit the image according to the text instruction in the image"
        },




        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/11.png",
            "prompt": "Follow the visual pointer and text to edit the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/12.png",
            "prompt": "Follow the visual pointer and text to edit the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/13.png",
            "prompt": "Follow the visual pointer and text to edit the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/14.png",
            "prompt": "Follow the visual pointer and text to edit the image"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/15.png",
            "prompt": "Follow the visual pointer and text to edit the image"
        },




        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/16.png",
            "prompt": "Modify the image at the annotated location according to the text instruction"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/17.png",
            "prompt": "Modify the image at the annotated location according to the text instruction"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/18.png",
            "prompt": "Modify the image at the annotated location according to the text instruction"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/19.png",
            "prompt": "Modify the image at the annotated location according to the text instruction"
        },
        {
            "img_path": "/home/disk2/hwj/LongCat-Image/type1/20.png",
            "prompt": "Modify the image at the annotated location according to the text instruction"
        },
        # 继续在这里添加更多字典...
    ]

    print(f"\n🚀 开始执行批量推理，共 {len(tasks)} 个任务...")

    # ==========================================
    # 6. 循环遍历任务进行推理并自动拼接文件名
    # ==========================================
    for i, task in enumerate(tasks):
        print(f"\n[{i + 1}/{len(tasks)}] 正在处理图片: {task['img_path']}")

        try:
            img = Image.open(task['img_path']).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法读取图片 {task['img_path']}，已跳过。错误信息: {e}")
            continue

        # 执行推理
        image = pipe(
            img,
            task['prompt'],
            negative_prompt='',
            guidance_scale=4.5,  # <--- TODO:这里可以考虑从 4.5 提高到如 7.5
            num_inference_steps=50,
            num_images_per_prompt=1,
            generator=torch.Generator("cpu").manual_seed(43)
        ).images[0]

        # 【核心修改】动态生成带有统一尾缀的输出路径
        # 例如: image_000000.JPEG -> image_000000
        original_basename = os.path.splitext(os.path.basename(task['img_path']))[0]
        # 拼接结果:
        final_out_path = os.path.join(OUTPUT_DIR, f"{original_basename}{UNIFIED_SUFFIX}.png")

        # 保存结果
        image.save(final_out_path)
        print(f"✅ 完成！结果已保存至 {final_out_path}")

    print("\n🎉 所有推理任务已全部完成！")