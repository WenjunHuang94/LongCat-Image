import os
import shutil
from tqdm import tqdm

# ================= 1. 配置区 =================
# 你现有的文件夹路径
# 结果图文件夹（目标生成图像 img_path）
RESULT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text-to-image-2M/output/"

# 原图文件夹（作为参考条件 ref_img_path）
ORIGIN_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text-to-image-2M/input/"

# 提取后存放的新文件夹路径（脚本会自动创建）
TARGET_BASE_DIR = "/storage/v-jinpewang/az_workspace/wenjun/LongCat-Image/text-to-image-2M-export_3000"
TARGET_RESULT_DIR = os.path.join(TARGET_BASE_DIR, "result_images")
TARGET_ORIGIN_DIR = os.path.join(TARGET_BASE_DIR, "origin_images")

# 需要提取的数量
MAX_SAMPLES = 3000


# ==========================================

def get_basename_dict(folder_path):
    """建立 {无后缀文件名: 完整绝对路径} 的映射，解决后缀不一致的问题"""
    mapping = {}
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    if not os.path.exists(folder_path):
        return mapping
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_exts:
            basename = os.path.splitext(filename)[0]
            mapping[basename] = os.path.join(folder_path, filename)
    return mapping


def main():
    # 创建目标文件夹
    os.makedirs(TARGET_RESULT_DIR, exist_ok=True)
    os.makedirs(TARGET_ORIGIN_DIR, exist_ok=True)

    print("正在扫描并配对图片...")
    origin_mapping = get_basename_dict(ORIGIN_DIR)
    result_mapping = get_basename_dict(RESULT_DIR)

    valid_count = 0

    # 获取所有能匹配上的基础名称列表
    matched_basenames = [name for name in result_mapping.keys() if name in origin_mapping]
    print(f"共找到 {len(matched_basenames)} 对匹配的图片。准备提取前 {MAX_SAMPLES} 对...")

    # 限制提取数量
    extract_list = matched_basenames[:MAX_SAMPLES]

    for basename in tqdm(extract_list, desc="复制文件中"):
        src_res = result_mapping[basename]
        src_ori = origin_mapping[basename]

        # 目标路径（保持原有文件名和后缀）
        dst_res = os.path.join(TARGET_RESULT_DIR, os.path.basename(src_res))
        dst_ori = os.path.join(TARGET_ORIGIN_DIR, os.path.basename(src_ori))

        # 复制文件
        shutil.copy2(src_res, dst_res)
        shutil.copy2(src_ori, dst_ori)

        valid_count += 1

    print(f"\n提取完成！成功复制了 {valid_count} 对图片。")
    print(f"它们存放在: {TARGET_BASE_DIR}")


if __name__ == "__main__":
    main()