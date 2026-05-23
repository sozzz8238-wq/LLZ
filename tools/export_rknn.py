"""
export_rknn.py — YOLOv8s-Pose → ONNX → RKNN 全流程转换

用法:
    python export_rknn.py

依赖:
    pip install ultralytics
    pip install rknn-toolkit2

步骤:
    1. yolov8s-pose.pt → yolov8s-pose.onnx (opset=12, imgsz=640)
    2. onnx → yolov8s-pose.rknn (RK3588 NPU 专用格式)
"""

import os
import sys

# 项目根目录下的 models/ 路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PT_PATH   = os.path.join(MODELS_DIR, "yolov8s-pose.pt")
ONNX_PATH = os.path.join(MODELS_DIR, "yolov8s-pose.onnx")
RKNN_PATH = os.path.join(MODELS_DIR, "yolov8s-pose.rknn")


def step1_pt_to_onnx():
    """步骤1: PyTorch → ONNX"""
    from ultralytics import YOLO

    print("[Step 1/2] 加载 PyTorch 模型...")
    model = YOLO(PT_PATH)

    print("[Step 1/2] 导出 ONNX (opset=12, imgsz=640)...")
    result = model.export(format="onnx", opset=12, imgsz=640)
    print(f"[Step 1/2] ONNX 导出完成 → {result}")


def step2_onnx_to_rknn():
    """步骤2: ONNX → RKNN (RK3588 NPU)"""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[WARN] rknn-toolkit2 未安装, 跳过 RKNN 转换")
        print("       板端部署前请执行: pip install rknn-toolkit2")
        return

    print("[Step 2/2] 正在转换 ONNX → RKNN...")
    rknn = RKNN(verbose=True)

    # 配置 RK3588 NPU
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )

    ret = rknn.load_onnx(model=ONNX_PATH)
    if ret != 0:
        print(f"[ERROR] ONNX 加载失败! ret={ret}")
        return

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"[ERROR] RKNN 构建失败! ret={ret}")
        return

    ret = rknn.export_rknn(RKNN_PATH)
    if ret == 0:
        print(f"[Step 2/2] RKNN 导出完成 → {RKNN_PATH}")
    else:
        print(f"[ERROR] RKNN 导出失败! ret={ret}")

    rknn.release()


def main():
    if not os.path.exists(PT_PATH):
        print(f"[ERROR] 找不到模型文件: {PT_PATH}")
        print("请将 yolov8s-pose.pt 放到 models/ 目录下")
        sys.exit(1)

    step1_pt_to_onnx()
    step2_onnx_to_rknn()

    print("\n[DONE] 模型已就绪, 请将 .rknn 文件部署到板端 models/ 目录")


if __name__ == "__main__":
    main()
