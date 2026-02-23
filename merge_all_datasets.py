import os
import shutil
import random
import json
from PIL import Image
from tqdm import tqdm

# ================= 1. 路径与配置区 =================
# 你的根目录
BASE_DIR = "/storage/v-jinpewang/az_workspace/wenjun/LongCat-Image"

# 四个数据源的映射关系字典
DATASET_MAPPING = {
    "generate": "text-to-image-2M-export_4000",
    "annotated_edit": "with_textbox-export_4000",
    "pointer_edit": "vismarked-merged-8000",
    "text": "wo_textbox_export_4000"
}

# 终极目标文件夹
TARGET_BASE_DIR = os.path.join(BASE_DIR, "merged_final_dataset")
TARGET_INPUT_DIR = os.path.join(TARGET_BASE_DIR, "input")
TARGET_OUTPUT_DIR = os.path.join(TARGET_BASE_DIR, "output")
INFO_TXT_PATH = os.path.join(TARGET_BASE_DIR, "final_train_data_info.txt")

# ================= 2. 提示词库 =================
PROMPTS = {
    "generate": [
        "根据图片文字描述绘画出真实图片", "根据文字描述生成真实图片", "按照文字描述绘制真实图片",
        "根据图片中的文字描述生成图片", "按照文字提示绘画出真实图片", "根据文字描述创作真实图片",
        "按照图片文字描述生成真实图像", "根据文字提示绘制真实图片", "按照文字描述生成图片",
        "根据图片中的文字描述绘画图片", "按照文字提示生成真实图片", "根据文字描述绘制图片",
        "按照图片文字描述创作真实图片", "根据文字提示生成图片", "按照文字描述绘画图片",
        "Generate a realistic image based on the text description in the image",
        "Draw a realistic image according to the text description",
        "Create a realistic image from the text description",
        "Generate an image based on the text prompt in the image",
        "Draw a realistic picture according to the text description",
        "Create a picture from the text description in the image",
    ],
    "annotated_edit": [
        "根据图片中的框标注和文字指令修改图像", "在图片中标注的指定位置添加文字描述的内容",
        "参考图中的颜色框标注，在对应位置生成目标物体", "按照标注框旁边的文字提示，修改图片中的指定区域",
        "根据标注指示，在图片对应位置进行绘画", "按照图中的框选区域和文字描述编辑图像",
        "根据图片中的标注框位置，绘画出文字描述的实景内容", "在图中框出的位置，按照文字指令进行修改",
        "结合图中的位置标注和文字提示，生成真实的场景",
        "Modify the image at the annotated location according to the text instruction",
        "Edit the specified area in the image based on the colored box and text prompt",
        "Add the object described by the text at the position indicated by the box",
        "Based on the annotations in the image, edit the specific region following the text",
        "Generate the content in the boxed area as described by the text prompt",
        "Follow the visual markers and text instructions to modify the image",
    ],
    "pointer_edit": [
        "根据图片中的文字指令编辑图像", "按照文字描述修改图片内容", "根据文字提示在图片上进行编辑",
        "Edit the image following the text description", "Modify the image based on the text prompt",
        "请识别图中的箭头指向，按照旁边的文字要求修改对应区域", "根据指示箭头和文字操作描述，对图片进行实景化修改",
        "根据图中箭头标记的位置，执行文字描述的编辑任务", "Follow the visual pointer and text to edit the image",
        "Execute the instruction written next to the arrow", "根据图片里的标注信息，把对应的物体换成文字描述的样子",
        "参考图中的提示文字和指向，完成图像编辑", "按照图片中的手写文字指令，对指定物体进行修改",
        "Look at the handwritten instructions in the image to perform the edit",
        "Based on the annotations, update the pointed part of the image"
    ],
    "text": [
        "根据图片中的文字指令编辑图像", "按照文字描述修改图片", "根据文字提示在图片上添加内容",
        "按照图片中的文字指令编辑图像", "根据文字描述编辑图片", "按照文字提示修改图像",
        "根据图片中的文字编辑图像", "按照文字指令在图片上添加内容", "根据文字描述在图片上进行编辑",
        "按照文字提示编辑图片", "根据图片中的文字指令修改图像", "按照文字描述在图片上添加元素",
        "根据文字提示编辑图像", "按照图片中的文字修改图像", "根据文字指令编辑图片",
        "Edit the image according to the text instruction in the image",
        "Modify the image based on the text description in the image",
        "Edit the image according to the text prompt", "Modify the image based on the text instruction",
        "Edit the image following the text description", "Apply the text instruction to edit the image",
        "Edit the image according to the text in the image", "Modify the image based on the text prompt in the image",
    ]
}


def main():
    # 创建目标文件夹
    os.makedirs(TARGET_INPUT_DIR, exist_ok=True)
    os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)

    total_processed = 0
    all_json_lines = []

    print("🚀 开始多模态数据终极大一统...\n")

    for task_type, folder_name in DATASET_MAPPING.items():
        print(f"📦 正在处理分类: [{task_type}] <- {folder_name}")

        source_input_dir = os.path.join(BASE_DIR, folder_name, "input")
        source_output_dir = os.path.join(BASE_DIR, folder_name, "output")

        if not os.path.exists(source_input_dir) or not os.path.exists(source_output_dir):
            print(f"  ❌ 错误：找不到对应的 input 或 output 文件夹，已跳过。")
            continue

        # 收集文件列表
        input_files = {f for f in os.listdir(source_input_dir) if not f.startswith('.')}
        output_files = {f for f in os.listdir(source_output_dir) if not f.startswith('.')}

        # 检查是否 1:1 一一对应
        matched_files = input_files.intersection(output_files)
        unmatched_inputs = input_files - output_files
        unmatched_outputs = output_files - input_files

        if unmatched_inputs or unmatched_outputs:
            print(f"  ⚠️ 警告：发现未配对的文件！")
            print(f"     缺少 Output 的 Input 文件数: {len(unmatched_inputs)}")
            print(f"     缺少 Input 的 Output 文件数: {len(unmatched_outputs)}")

        print(f"  ✅ 完美匹配数量: {len(matched_files)}")

        # 设定固定随机种子（保证生成的 prompt 具有可复现性）
        random.seed(42)

        # 复制文件并生成标注
        for filename in tqdm(matched_files, desc=f"  打包 {task_type}", leave=False):
            # 原始路径
            src_in = os.path.join(source_input_dir, filename)
            src_out = os.path.join(source_output_dir, filename)

            # 强力防重名机制：加上 task_type 前缀 (例如 generate_001.png)
            safe_filename = f"{task_type}_{filename}"
            dst_in = os.path.join(TARGET_INPUT_DIR, safe_filename)
            dst_out = os.path.join(TARGET_OUTPUT_DIR, safe_filename)

            # 复制
            shutil.copy2(src_in, dst_in)
            shutil.copy2(src_out, dst_out)

            # 获取图片宽高信息 (以防有不同分辨率的数据，动态获取最安全)
            with Image.open(dst_out) as img:
                width, height = img.size

            # 随机选取对应的指令
            prompt = random.choice(PROMPTS[task_type])

            # 构建标准的 jsonl 行 (与 Diffusers/LongCat 要求一致)
            # 注意：img_path 是生成的图(output), ref_img_path 是条件原图(input)
            info_dict = {
                "img_path": dst_out,
                "ref_img_path": dst_in,
                "prompt": prompt,
                "width": width,
                "height": height
            }

            all_json_lines.append(info_dict)
            total_processed += 1

    # 将所有的信息打乱，保证训练时 DataLoader 混合得更均匀
    print("\n🔀 正在打乱所有数据条目的顺序...")
    random.shuffle(all_json_lines)

    # 写入最终的 train_data_info.txt
    print(f"📝 正在生成训练标注文件: {INFO_TXT_PATH}")
    with open(INFO_TXT_PATH, "w", encoding="utf-8") as f:
        for line in all_json_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("\n🎉 大一统圆满完成！")
    print(f"📂 所有文件统一存放在: {TARGET_BASE_DIR}")
    print(f"📊 最终共计生成 {total_processed} 条训练数据。")
    print(f"✨ 你的 Stage 2 数据集已经完美就位，随时可以开始炼丹！")


if __name__ == "__main__":
    main()