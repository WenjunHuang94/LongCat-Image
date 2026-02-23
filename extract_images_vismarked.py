import os
import shutil
import random
from tqdm import tqdm

# ================= 1. 配置区 =================
# 你现有的 8 个子文件夹所在的根目录
BASE_DATA_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/"

# 需要遍历的数据集列表
DATASETS = [
    "omniedit_attribute_modification",
    "omniedit_object_swap",
    "omniedit_removal",
    "omniedit_swap",
    "ultraedit_change_color",
    "ultraedit_change_local",
    "ultraedit_replace",
    "ultraedit_turn"
]

# 最终大一统存放的新文件夹路径（自动创建）
TARGET_BASE_DIR = "/storage/v-jinpewang/az_workspace/wenjun/LongCat-Image/vismarked-merged-8000"
TARGET_ORIGIN_DIR = os.path.join(TARGET_BASE_DIR, "input")
TARGET_RESULT_DIR = os.path.join(TARGET_BASE_DIR, "output")

# 每个子文件夹需要提取的数量
SAMPLES_PER_DATASET = 1000


# ==========================================

def get_id_mapping(folder_path, suffix_to_remove=""):
    """提取纯ID (xxx) 并映射到完整路径，支持无后缀直接提取"""
    mapping = {}
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    if not os.path.exists(folder_path):
        print(f"❌ 警告：找不到文件夹 {folder_path}")
        return mapping

    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)
        if ext.lower() in valid_exts:
            # 如果有特定后缀需要去掉 (保留之前的兼容性)
            if suffix_to_remove and name.endswith(suffix_to_remove):
                pure_id = name[:-len(suffix_to_remove)]
                mapping[pure_id] = os.path.join(folder_path, filename)
            # 如果没有特定后缀，直接用文件名作为 ID
            elif not suffix_to_remove:
                mapping[name] = os.path.join(folder_path, filename)

    return mapping


def main():
    # 1. 创建终极目标文件夹
    os.makedirs(TARGET_ORIGIN_DIR, exist_ok=True)
    os.makedirs(TARGET_RESULT_DIR, exist_ok=True)

    total_extracted = 0

    print("🚀 开始多数据源混合抽取大业...\n")

    # 2. 遍历每一个子数据集
    for dataset_name in DATASETS:
        print(f"📦 正在处理数据集: {dataset_name}")

        # 拼凑当前数据集的 input 和 output 路径
        current_input_dir = os.path.join(BASE_DATA_DIR, dataset_name, "input")
        current_output_dir = os.path.join(BASE_DATA_DIR, dataset_name, "output")

        # 这里假设这 8 个数据集的文件名就是纯粹的一一对应（如 001.png 对 001.png）
        # 如果它们也有特定的后缀，可以把 "" 改成 "_textbox" 等
        origin_mapping = get_id_mapping(current_input_dir, suffix_to_remove="")
        result_mapping = get_id_mapping(current_output_dir, suffix_to_remove="")

        # 找出两边都有的纯 ID
        matched_ids = [img_id for img_id in result_mapping.keys() if img_id in origin_mapping]

        if len(matched_ids) == 0:
            print(f"  ⚠️ 跳过 {dataset_name}：未找到任何配对的图片。\n")
            continue

        # 确定实际提取数量（防止某些文件夹不够 1000 张报错）
        actual_samples = min(SAMPLES_PER_DATASET, len(matched_ids))

        # 随机抽取
        random.seed(42)  # 固定种子
        extract_list = random.sample(matched_ids, actual_samples)

        dataset_count = 0

        # 3. 开始复制并重命名
        for img_id in tqdm(extract_list, desc=f"  复制 {dataset_name} 中", leave=False):
            src_ori = origin_mapping[img_id]
            src_res = result_mapping[img_id]

            ext_ori = os.path.splitext(src_ori)[1]
            ext_res = os.path.splitext(src_res)[1]

            # 【核心修改】将 数据集名称 作为前缀加入，彻底杜绝重名！
            # 生成的文件名例如: omniedit_swap_0001.png
            new_filename_ori = f"{dataset_name}_{img_id}{ext_ori}"
            new_filename_res = f"{dataset_name}_{img_id}{ext_res}"

            dst_ori = os.path.join(TARGET_ORIGIN_DIR, new_filename_ori)
            dst_res = os.path.join(TARGET_RESULT_DIR, new_filename_res)

            shutil.copy2(src_ori, dst_ori)
            shutil.copy2(src_res, dst_res)

            dataset_count += 1
            total_extracted += 1

        print(f"  ✅ 成功抽取 {dataset_count} 对\n")

    print(f"🎉 全部混合抽取完成！")
    print(f"📂 大一统文件夹位于: {TARGET_BASE_DIR}")
    print(f"📊 总计成功提取: {total_extracted} 对图片 (Input/Output 绝对一一对应)")


if __name__ == "__main__":
    main()