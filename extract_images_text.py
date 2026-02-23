import os
import shutil
import random
from tqdm import tqdm

# ================= 1. 配置区 =================
# 你现有的 wo_textbox 根目录
BASE_DATA_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/"

# 最终大一统存放的新文件夹路径（自动创建）
TARGET_BASE_DIR = "/storage/v-jinpewang/az_workspace/wenjun/LongCat-Image/wo_textbox_export_4000"
TARGET_ORIGIN_DIR = os.path.join(TARGET_BASE_DIR, "input")
TARGET_RESULT_DIR = os.path.join(TARGET_BASE_DIR, "output")

# 总共期望提取的数量
TARGET_TOTAL_SAMPLES = 5000


# ==========================================

def get_id_mapping(folder_path):
    """提取纯ID映射到完整路径 (针对 wo_textbox，假设 input/output 文件名纯粹一致)"""
    mapping = {}
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    if not os.path.exists(folder_path):
        return mapping

    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)
        if ext.lower() in valid_exts:
            mapping[name] = os.path.join(folder_path, filename)

    return mapping


def main():
    # 1. 扫描所有的任务子文件夹 (共 16 个)
    task_folders = [f for f in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, f))]
    task_folders.sort()

    print(f"🔍 扫描到 {len(task_folders)} 个任务子文件夹。开始统计配对数据...")

    # 存储每个文件夹的统计信息
    dataset_info = {}
    total_available_pairs = 0

    # 2. 深入统计每个任务的情况
    for task in task_folders:
        task_path = os.path.join(BASE_DATA_DIR, task)

        # 自动寻找 ultraedit 或 omniedit 文件夹
        source_dir_name = None
        for sub in os.listdir(task_path):
            if sub in ['ultraedit', 'omniedit'] and os.path.isdir(os.path.join(task_path, sub)):
                source_dir_name = sub
                break

        if not source_dir_name:
            print(f"  ⚠️ 跳过 {task}: 未找到 ultraedit 或 omniedit 文件夹。")
            continue

        input_dir = os.path.join(task_path, source_dir_name, "input")
        output_dir = os.path.join(task_path, source_dir_name, "output")

        origin_mapping = get_id_mapping(input_dir)
        result_mapping = get_id_mapping(output_dir)

        # 找出配对的图片
        matched_ids = [img_id for img_id in result_mapping.keys() if img_id in origin_mapping]

        dataset_info[task] = {
            'matched_ids': matched_ids,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'source_type': source_dir_name
        }

        total_available_pairs += len(matched_ids)
        print(f"  📊 {task} ({source_dir_name}): 找到 {len(matched_ids)} 对匹配数据。")

    print(f"\n✅ 统计完毕！这 {len(dataset_info)} 个有效任务中，总计共有 {total_available_pairs} 对完美匹配的图片。")

    # 3. 计算每个任务的分配额度
    valid_task_count = len(dataset_info)
    if valid_task_count == 0:
        print("❌ 没有找到任何有效数据，程序退出。")
        return

    # 计算均摊配额 (4000 / 16 = 250)
    quota_per_task = TARGET_TOTAL_SAMPLES // valid_task_count
    print(f"\n⚖️ 均衡采样策略：目标总量 {TARGET_TOTAL_SAMPLES}，共 {valid_task_count} 个任务。")
    print(f"⚖️ 每个任务计划抽取上限：{quota_per_task} 对图。")

    # 4. 创建目标文件夹并开始抽取
    os.makedirs(TARGET_ORIGIN_DIR, exist_ok=True)
    os.makedirs(TARGET_RESULT_DIR, exist_ok=True)

    total_extracted = 0
    print("\n🚀 开始抽取并重命名...")

    for task, info in dataset_info.items():
        matched_ids = info['matched_ids']

        # 决定抽取数量：如果不满配额，就全拿；如果超过配额，就随机抽取配额数量
        actual_samples = min(quota_per_task, len(matched_ids))

        if actual_samples == 0:
            continue

        random.seed(42)  # 保证每次抽取的结果一致
        extract_list = random.sample(matched_ids, actual_samples)

        origin_mapping = get_id_mapping(info['input_dir'])
        result_mapping = get_id_mapping(info['output_dir'])

        # 开始复制
        for img_id in tqdm(extract_list, desc=f"  打包 {task}", leave=False):
            src_ori = origin_mapping[img_id]
            src_res = result_mapping[img_id]

            ext_ori = os.path.splitext(src_ori)[1]
            ext_res = os.path.splitext(src_res)[1]

            # 【核心隔离机制】前缀包含任务名和数据源，彻底防重名
            # 例如: change_color_ultraedit_0001.png
            new_filename_ori = f"{task}_{info['source_type']}_{img_id}{ext_ori}"
            new_filename_res = f"{task}_{info['source_type']}_{img_id}{ext_res}"

            dst_ori = os.path.join(TARGET_ORIGIN_DIR, new_filename_ori)
            dst_res = os.path.join(TARGET_RESULT_DIR, new_filename_res)

            shutil.copy2(src_ori, dst_ori)
            shutil.copy2(src_res, dst_res)

            total_extracted += 1

    print(f"\n🎉 大一统混合抽取完成！")
    print(f"📂 最终文件夹位于: {TARGET_BASE_DIR}")
    print(f"📊 实际成功提取并混合: {total_extracted} 对图片。")


if __name__ == "__main__":
    main()