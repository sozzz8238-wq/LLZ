import os
import shutil

def classify_files_by_prefix(source_dir, output_dir):
    """
    按文件名前缀 m01-m10 分类复制文件
    :param source_dir: 源文件夹路径（存放所有m01-m10文件）
    :param output_dir: 输出文件夹路径（分类后保存）
    """
    # 1. 检查源文件夹是否存在
    if not os.path.isdir(source_dir):
        print(f"❌ 错误：源文件夹不存在 → {source_dir}")
        return

    # 2. 定义需要匹配的前缀 m01 ~ m10
    target_prefixes = [f"m{i:02d}" for i in range(1, 11)]
    print(f"✅ 准备匹配文件前缀：{target_prefixes}")

    # 3. 遍历源文件夹所有文件
    file_count = 0
    for filename in os.listdir(source_dir):
        # 跳过文件夹，只处理文件
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue

        # 匹配文件名开头的 m01-m10
        matched_prefix = None
        for prefix in target_prefixes:
            if filename.startswith(prefix):
                matched_prefix = prefix
                break

        # 未匹配到则跳过
        if not matched_prefix:
            continue

        # 4. 创建输出子文件夹（输出目录/m01、m02...）
        target_sub_dir = os.path.join(output_dir, matched_prefix)
        os.makedirs(target_sub_dir, exist_ok=True)  # 自动创建，已存在不报错

        # 5. 复制文件到目标子文件夹（安全复制，不修改源文件）
        target_file_path = os.path.join(target_sub_dir, filename)
        shutil.copy2(file_path, target_file_path)  # copy2 保留文件属性

        file_count += 1
        print(f"📄 已复制：{filename} → {target_sub_dir}")

    print(f"\n🎉 任务完成！共复制 {file_count} 个文件")
    print(f"📂 文件已分类保存至：{output_dir}")

if __name__ == '__main__':
    # ====================== 【仅需修改这2个路径】 ======================
    SOURCE_FOLDER = r"E:\嵌赛\UI-PRMD_data\Segmented Movements\Kinect\Angles"    # 你的源文件夹（所有m01-m10文件所在）
    OUTPUT_FOLDER = r"E:\嵌赛\UI-PRMD_data\Segmented Movements\Kinect\categories_Angles"       # 输出文件夹（自动创建m01-m10子文件夹）
    # =================================================================

    # 执行分类复制
    classify_files_by_prefix(SOURCE_FOLDER, OUTPUT_FOLDER)