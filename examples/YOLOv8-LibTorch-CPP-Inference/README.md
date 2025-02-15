# YOLOv8 LibTorch Inference C++

This example demonstrates how to perform inference using YOLOv8 models in C++ with LibTorch API.

## Dependencies

| Dependency   | Version  |
| ------------ | -------- |
| OpenCV       | >=4.0.0  |
| C++ Standard | >=17     |
| Cmake        | >=3.18   |
| Libtorch     | >=1.12.1 |

## Usage

```bash
git clone ultralytics
cd ultralytics
pip install .
cd examples/YOLOv8-LibTorch-CPP-Inference

mkdir build
cd build
cmake ..
make
./yolov8_libtorch_inference
```

## Exporting YOLOv8

To export YOLOv8 models:

```commandline
yolo export model=yolov8s.pt imgsz=640 format=torchscript
```

## Maintainer

This project is maintained by Deepak Rambarki, an AI Engineer specializing in high-performance machine learning deployments. With a strong background in PyTorch, C++, and scalable data systems, Deepak focuses on bridging the gap between research models and production-ready C++ implementations.

Contact Information:
- Email: rambarkideepak@gmail.com
- Role: AI Engineer
- Expertise: PyTorch, C++, Python, Computer Vision