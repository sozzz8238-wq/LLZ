from ultralytics import YOLO


def main():
    # 第一次运行会自动下载 yolov8n-pose.pt (大约 6MB)
    print("正在加载官方原生模型...")
    model = YOLO("yolov8s-pose.pt")

    # 直接导出为 ONNX，锁定输入尺寸 640x640，算子集 12
    print("正在转换为 ONNX 格式...")
    success = model.export(format="onnx", opset=12, imgsz=640)

    print(f"✅ 转换完成！你的武器保存在: {success}")


if __name__ == '__main__':
    main()