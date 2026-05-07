import numpy as np
import json
import glob
import os

# ==========================================
# 核心算法：直接从欧拉角中提取弯曲主轴
# ==========================================
def extract_angle_from_euler(txt_file_path):
    axis_data = [[], [], []]

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = [float(x) for x in line.strip().split(',')]
            if len(data) != 66:
                continue

            # 索引 45,46,47 是左膝盖的欧拉角
            axis_data[0].append(data[45])
            axis_data[1].append(data[46])
            axis_data[2].append(data[47])

    if len(axis_data[0]) == 0:
        return []

    # 取方差最大的轴为弯曲主轴
    variances = [np.var(axis_data[0]), np.var(axis_data[1]), np.var(axis_data[2])]
    bending_axis = np.argmax(variances)
    raw_curve = axis_data[bending_axis]

    # 角度映射
    mapped_curve = [180.0 - abs(angle) for angle in raw_curve]
    return mapped_curve

# ==========================================
# 主干：批量读取与时间归一化融合
# ==========================================
def build_ultimate_template(input_folder, action_name, output_json):
    search_pattern = os.path.join(input_folder, "*_angles.txt")
    file_list = glob.glob(search_pattern)
    file_list = [f for f in file_list if "_inc" not in f]

    if not file_list:
        print(f"❌ 错误：{input_folder} 无有效数据，跳过")
        return

    print(f"📂 正在处理 [{action_name}] ...")
    print(f"   找到 {len(file_list)} 个受试者数据")

    TARGET_LENGTH = 100
    normalized_curves = []

    for file in file_list:
        raw_curve = extract_angle_from_euler(file)
        if len(raw_curve) < 10:
            continue

        # 时间归一化插值
        old_x = np.linspace(0, 1, len(raw_curve))
        new_x = np.linspace(0, 1, TARGET_LENGTH)
        resampled_curve = np.interp(new_x, old_x, raw_curve)
        normalized_curves.append(resampled_curve)

    if not normalized_curves:
        print(f"❌ {action_name} 无有效曲线，跳过\n")
        return

    # 生成平均曲线
    curves_matrix = np.array(normalized_curves)
    golden_curve = np.mean(curves_matrix, axis=0)

    golden_data = {
        "action_name": action_name,
        "data_sources": len(normalized_curves),
        "normalized_length": TARGET_LENGTH,
        "angle_sequence": [round(num, 2) for num in golden_curve.tolist()]
    }

    # 自动创建输出文件夹
    os.makedirs("golden_templates", exist_ok=True)
    with open(output_json, 'w') as out_file:
        json.dump(golden_data, out_file, indent=4)

    print(f"✅ {action_name} 模板生成完成: {output_json}\n")

# ==========================================
# 【批量处理主函数】自动遍历所有动作文件夹
# ==========================================
def batch_build_all_templates(root_kinect_folder):
    """
    批量处理 Kinect 根目录下的所有动作文件夹（m01/m02/m03...）
    :param root_kinect_folder: Kinect 总文件夹路径
    """
    # 1. 获取根目录下所有子文件夹
    action_folders = []
    for item in os.listdir(root_kinect_folder):
        item_path = os.path.join(root_kinect_folder, item)
        # 只保留 文件夹 + 以m开头的动作文件夹（m01/m02...）
        if os.path.isdir(item_path) and item.startswith("m"):
            action_folders.append((item, item_path))

    if not action_folders:
        print("❌ 未找到任何动作文件夹！")
        return

    print(f"🚀 找到 {len(action_folders)} 个动作，开始批量处理\n")

    # 2. 循环处理每个动作
    for action_code, folder_path in action_folders:
        # 自动生成动作名：m01_Squat / m02_XXX ...
        action_name = f"{action_code}_Action"
        # 自动生成输出json路径
        output_json = f"golden_templates/{action_code}_golden.json"
        # 执行生成
        build_ultimate_template(folder_path, action_name, output_json)

    print("🎉 所有动作模板批量生成完成！")

if __name__ == '__main__':
    # ====================== 只需要改这一个路径！======================
    # 填写 Kinect 总根目录（包含 m01/m02/m03... 所有文件夹）
    KINECT_ROOT_FOLDER = r"E:\嵌赛\UI-PRMD_data\Segmented Movements\Kinect\categories_Angles"
    # ==============================================================

    # 一键批量处理所有动作
    batch_build_all_templates(KINECT_ROOT_FOLDER)