import os
import shutil
import random
from tqdm import tqdm

# ================= 1. 配置区 =================
# 结果图文件夹（目标生成图像 img_path）
RESULT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_Accgen/with_textbox/output/"

# 原图文件夹（作为参考条件 ref_img_path）
ORIGIN_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_Accgen/with_textbox/input/"

# 提取后存放的新文件夹路径（自动创建）
TARGET_BASE_DIR = "/storage/v-jinpewang/az_workspace/wenjun/LongCat-Image/with_textbox-export_4000"
TARGET_ORIGIN_DIR = os.path.join(TARGET_BASE_DIR, "input")
TARGET_RESULT_DIR = os.path.join(TARGET_BASE_DIR, "output")

# 需要提取的数量
MAX_SAMPLES = 4000


# ==========================================

def get_id_mapping(folder_path, suffix_to_remove):
    """提取纯ID (xxx) 并映射到完整路径，解决文件名后缀不同的问题"""
    mapping = {}
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到文件夹 {folder_path}")
        return mapping

    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)
        # 确保是图片，并且以指定的后缀结尾
        if ext.lower() in valid_exts and name.endswith(suffix_to_remove):
            # 去掉特定后缀（例如 "_textbox"），只保留 "xxx"
            pure_id = name[:-len(suffix_to_remove)]
            mapping[pure_id] = os.path.join(folder_path, filename)

    return mapping


def main():
    # 创建目标文件夹
    os.makedirs(TARGET_ORIGIN_DIR, exist_ok=True)
    os.makedirs(TARGET_RESULT_DIR, exist_ok=True)

    print("正在扫描并配堆图片...")
    # 分别提取去掉后缀后的 xxx 作为字典的 key
    origin_mapping = get_id_mapping(ORIGIN_DIR, "_textbox")
    result_mapping = get_id_mapping(RESULT_DIR, "_edited")

    # 找出两边都有的纯 ID (xxx)
    matched_ids = [img_id for img_id in result_mapping.keys() if img_id in origin_mapping]
    print(f"共找到 {len(matched_ids)} 对完美匹配的图片。")

    if len(matched_ids) < MAX_SAMPLES:
        print(f"⚠️ 警告：匹配的数量（{len(matched_ids)}）少于目标数量（{MAX_SAMPLES}）！将提取所有匹配项。")
        extract_list = matched_ids
    else:
        # 强烈建议：打乱顺序随机抽取，保证数据多样性
        random.seed(42)  # 固定随机种子，保证每次跑抽取的 4000 张都是一样的，方便复现
        extract_list = random.sample(matched_ids, MAX_SAMPLES)
        print(f"已随机抽取 {len(extract_list)} 对图片准备提取...")

    valid_count = 0
    # 开始复制并重命名
    for img_id in tqdm(extract_list, desc="复制并重命名文件中"):
        src_ori = origin_mapping[img_id]
        src_res = result_mapping[img_id]

        # 获取原文件的扩展名（例如 .png）
        ext_ori = os.path.splitext(src_ori)[1]
        ext_res = os.path.splitext(src_res)[1]

        # 【核心修改】新文件名统一为纯 ID (例如: xxx.png)，彻底去掉 _textbox 和 _edited
        dst_ori = os.path.join(TARGET_ORIGIN_DIR, f"{img_id}{ext_ori}")
        dst_res = os.path.join(TARGET_RESULT_DIR, f"{img_id}{ext_res}")

        shutil.copy2(src_ori, dst_ori)
        shutil.copy2(src_res, dst_res)
        valid_count += 1

    print(f"\n✅ 提取并重命名完成！成功处理了 {valid_count} 对图片。")
    print(f"📂 它们存放在: {TARGET_BASE_DIR}")
    print(f"📄 文件名已统一格式，例如: input/xxx.png 对应 output/xxx.png")


if __name__ == "__main__":
    main()