import os
import json
from PIL import Image
from tqdm import tqdm

# 解除大图片限制，防止某些异常大图导致报错
Image.MAX_IMAGE_PIXELS = None

# ================= 1. 配置区 =================
# 结果图文件夹（目标生成图像 img_path）
RESULT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text-to-image-2M/output/"

# 原图文件夹（作为参考条件 ref_img_path）
ORIGIN_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text-to-image-2M/input/"

# 最终生成的 txt (JSONL) 文件保存路径
OUTPUT_TXT = "data_example/text-to-image-2M-3000.txt"

# 你固定的 Prompt 指令
FIXED_PROMPT = "Generate a realistic image based on the text description in the image"

# 【新增】最大生成数量限制（设置为你想截取的数量，如果要全量处理，可以设为 float('inf')）
MAX_SAMPLES = 3000


# ==========================================

def get_basename_dict(folder_path):
    """
    扫描文件夹，建立 {无后缀文件名: 完整绝对路径} 的映射字典
    """
    mapping = {}
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    if not os.path.exists(folder_path):
        print(f"警告：找不到文件夹 {folder_path}")
        return mapping

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_exts:
            basename = os.path.splitext(filename)[0]
            mapping[basename] = os.path.join(folder_path, filename)

    return mapping


def main():
    print("正在扫描并建立原始图片索引...")
    origin_mapping = get_basename_dict(ORIGIN_DIR)

    print("正在扫描结果图片...")
    result_mapping = get_basename_dict(RESULT_DIR)

    if not result_mapping:
        print("没有找到任何结果图片，请检查 RESULT_DIR 路径是否正确！")
        return

    print(f"发现 {len(result_mapping)} 张结果图，{len(origin_mapping)} 张原始图。")
    print(f"目标提取数量: {MAX_SAMPLES} 张，开始匹配并提取分辨率...")

    valid_count = 0
    missing_count = 0
    error_count = 0

    # 打开目标 txt 文件准备写入
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        # 使用自定义总量的进度条
        pbar = tqdm(total=min(MAX_SAMPLES, len(result_mapping)), desc="处理进度")

        # 遍历结果图
        for basename, res_path in result_mapping.items():
            # 【核心逻辑】如果已经达到限制数量，直接跳出循环结束运行
            if valid_count >= MAX_SAMPLES:
                break

            # 1. 查找对应的原图
            if basename not in origin_mapping:
                missing_count += 1
                continue

            ori_path = origin_mapping[basename]

            # 2. 读取图片宽高
            try:
                with Image.open(res_path) as img:
                    w, h = img.size

                # 3. 组装符合 DataLoader 要求的字典
                data_dict = {
                    "img_path": res_path,
                    "ref_img_path": ori_path,
                    "prompt": FIXED_PROMPT,
                    "width": w,
                    "height": h
                }

                # 4. 转换为 JSON 字符串并写入
                f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')

                # 成功处理一条，计数器+1，更新进度条
                valid_count += 1
                pbar.update(1)

            except Exception as e:
                error_count += 1
                print(f"\n读取图片失败，已跳过: {res_path} | 错误信息: {e}")

        pbar.close()

    # 打印最终统计信息
    print("\n" + "=" * 40)
    print(f"数据处理完成！")
    print(f"成功生成有效数据: {valid_count} 条")
    if valid_count == MAX_SAMPLES:
        print(f"已达到预设的 {MAX_SAMPLES} 条限制，自动停止。")
    print(f"缺少对应原图跳过: {missing_count} 条")
    print(f"图片损坏读取失败: {error_count} 条")
    print(f"结果已保存至: {OUTPUT_TXT}")
    print("=" * 40)


if __name__ == "__main__":
    main()