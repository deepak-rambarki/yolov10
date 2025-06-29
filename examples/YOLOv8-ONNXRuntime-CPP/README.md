# YOLOv8 OnnxRuntime C++

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white">

This repository provides a high-performance implementation for performing inference using YOLOv8 in C++ with ONNX Runtime and OpenCV.

## Benefits

- Optimized for deployment in industrial environments.
- Faster than OpenCV's DNN inference on both CPU and GPU.
- Supports FP32 and FP16 CUDA acceleration.

## Note

1. Benefiting from the latest Ultralytics release, a Transpose op is added to the YOLOv8 model, ensuring v8 and v5 have the same output shape. Consequently, this project supports inference for YOLOv5, YOLOv7, and YOLOv8.

## Exporting YOLOv8 Models

To export YOLOv8 models for use with this project, use the following Python script:

```python
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
```

Alternatively, you can use the following command in the terminal:

```bash
yolo export model=yolov8n.pt opset=12 simplify=True dynamic=False format=onnx imgsz=640,640
```

## Exporting YOLOv8 FP16 Models

```python
import onnx
from onnxconverter_common import float16

model = onnx.load(R'YOUR_ONNX_PATH')
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, R'YOUR_FP16_ONNX_PATH')
```

## Download COCO.yaml file

To run the examples, you will need the coco.yaml file. You can download it manually from the official Ultralytics repository or via the following link: [coco.yaml](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml)

## Dependencies

| Dependency                       | Version        |
| -------------------------------- | -------------- |
| Onnxruntime(linux,windows,macos) | >=1.14.1       |
| OpenCV                           | >=4.0.0        |
| C++ Standard                     | >=17           |
| Cmake                            | >=3.5          |
| Cuda (Optional)                  | >=11.4 <12.0   |
| cuDNN (Cuda required)            | =8             |

Note: The dependency on C++17 is required for the filesystem features.

Note (2): Due to ONNX Runtime requirements, CUDA 11 and cuDNN 8 are currently necessary. These requirements may be updated in future releases.

## Build

1. Clone the repository to your local machine.

2. Navigate to the root directory of the repository.

3. Create a build directory and navigate to it:

    ```console
    mkdir build && cd build
    ```

4. Run CMake to generate the build files:

    ```console
    cmake ..
    ```

5. Build the project:

    ```console
    make
    ```

6. The built executable will be located in the build directory.

## Usage

```c++
// Configure parameters as needed
// Ensure the device and model type (fp32 or fp16) match your environment
DL_INIT_PARAM params;
params.rectConfidenceThreshold = 0.1;
params.iouThreshold = 0.5;
params.modelPath = "yolov8n.onnx";
params.imgSize = { 640, 640 };
params.cudaEnable = true;
params.modelType = YOLO_DETECT_V8;
yoloDetector->CreateSession(params);
Detector(yoloDetector);
```

## Maintainer

Deepak Rambarki is an AI Engineer with extensive experience in developing scalable machine learning pipelines and high-performance computer vision solutions. With a focus on bridging the gap between research and industrial deployment, he maintains this repository to provide efficient C++ inference tools for the AI community.

Contact: rambarkideepak@gmail.com