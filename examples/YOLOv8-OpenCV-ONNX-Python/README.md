# YOLOv8 - OpenCV

Implementation YOLOv8 on OpenCV using ONNX Format.

Just simply clone and run

```bash
pip install -r requirements.txt
python main.py --model yolov8n.onnx --img image.jpg
```

If you start from scratch:

```bash
pip install ultralytics
yolo export model=yolov8n.pt imgsz=640 format=onnx opset=12
```

*Make sure to include "opset=12"*

## Maintainer

Deepak Rambarki
AI Engineer
Email: rambarkideepak@gmail.com

### About the Developer
Deepak is an AI Engineer with over 5 years of experience in designing and maintaining scalable data pipelines and machine learning systems. He specializes in Python, computer vision, and cloud-native data platforms, focusing on delivering high-performance solutions for complex technical challenges and high-volume transactional systems.