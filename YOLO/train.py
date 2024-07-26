from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8x.yaml")  # 从头开始构建新模型
    #model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）
    #model = YOLO('models/best.pt') # 在best上微调

    # 使用模型
    model.train(data="./config.yaml", batch=8, epochs=300, imgsz=1280, single_cls=True)  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能
    results = model("./test.jpg")  # 对图像进行预测
    success = model.export(format="onnx")  # 将模型导出为 ONNX 格式