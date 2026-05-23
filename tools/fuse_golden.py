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
        print(f"ERROR: {input_folder} no valid data, skipping")
        return

    print(f"Processing [{action_name}] ...")
    print(f"   Found {len(file_list)} subject data files")

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
        print(f"{action_name} no valid curve, skipping\n")
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

    # 输出到项目根目录下的 templates/
    os.makedirs(output_json, exist_ok=True) if os.path.isdir(output_json) else os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as out_file:
        json.dump(golden_data, out_file, indent=4)

    print(f"DONE {action_name} template saved: {output_json}\n")

# ==========================================
# 批量处理主函数
# ==========================================
def batch_build_all_templates(root_kinect_folder, output_dir):
    action_folders = []
    for item in os.listdir(root_kinect_folder):
        item_path = os.path.join(root_kinect_folder, item)
        if os.path.isdir(item_path) and item.startswith("m"):
            action_folders.append((item, item_path))

    if not action_folders:
        print("ERROR: No action folders found!")
        return

    print(f"Found {len(action_folders)} actions, start batch processing\n")

    os.makedirs(output_dir, exist_ok=True)

    for action_code, folder_path in action_folders:
        action_name = f"{action_code}_Action"
        output_json = os.path.join(output_dir, f"{action_code}_golden.json")
        build_ultimate_template(folder_path, action_name, output_json)

    print("All action templates generated!")

if __name__ == '__main__':
    # ====================== Configuration ======================
    KINECT_ROOT_FOLDER = r"E:\UI-PRMD_data\Segmented Movements\Kinect\categories_Angles"
    # 输出到项目 templates/ 目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "templates")
    # ============================================================

    batch_build_all_templates(KINECT_ROOT_FOLDER, OUTPUT_DIR)
