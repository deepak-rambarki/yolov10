# YOLOv8-ONNXRuntime-Rust for All the Key YOLO Tasks

This repository provides a Rust implementation for performing YOLOv8 tasks including Classification, Segmentation, Detection, and Pose Detection using ONNXRuntime. It is designed for high-performance inference across various hardware accelerators.

## Features

- Support for Classification, Segmentation, Detection, and Pose (Keypoints) Detection tasks.
- Support for FP16 and FP32 ONNX models.
- Support for CPU, CUDA, and TensorRT execution providers to accelerate computation.
- Support for dynamic input shapes (batch, width, height).

## Installation

### 1. Install Rust

Please follow the Rust official installation guide: https://www.rust-lang.org/tools/install

### 2. Install ONNXRuntime

This repository uses the "ort" crate, which is an ONNXRuntime wrapper for Rust: https://docs.rs/ort/latest/ort/

You can follow the instructions in the "ort" documentation or perform the following steps:

- Step 1: Download ONNXRuntime (https://github.com/microsoft/onnxruntime/releases)
- Step 2: Set the environment variable PATH for linking.

On Ubuntu, you can configure it as follows:

```bash
vim ~/.bashrc

# Add the path of ONNXRuntime lib
export LD_LIBRARY_PATH=/home/user/Documents/onnxruntime-linux-x64-gpu-1.16.3/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc
```

### 3. [Optional] Install CUDA, CuDNN, and TensorRT

- CUDA execution provider requires CUDA v11.6+.
- TensorRT execution provider requires CUDA v11.4+ and TensorRT v8.4+.

## Get Started

### 1. Export the YOLOv8 ONNX Models

```bash
pip install -U ultralytics

# export onnx model with dynamic shapes
yolo export model=yolov8m.pt format=onnx simplify dynamic
yolo export model=yolov8m-cls.pt format=onnx simplify dynamic
yolo export model=yolov8m-pose.pt format=onnx simplify dynamic
yolo export model=yolov8m-seg.pt format=onnx simplify dynamic

# export onnx model with constant shapes
yolo export model=yolov8m.pt format=onnx simplify
yolo export model=yolov8m-cls.pt format=onnx simplify
yolo export model=yolov8m-pose.pt format=onnx simplify
yolo export model=yolov8m-seg.pt format=onnx simplify
```

### 2. Run Inference

This will perform inference with the ONNX model on the source image.

```bash
cargo run --release -- --model <MODEL> --source <SOURCE>
```

Set --cuda to use the CUDA execution provider to speed up inference:

```bash
cargo run --release -- --cuda --model <MODEL> --source <SOURCE>
```

Set --trt to use the TensorRT execution provider. You can also set --fp16 to use the TensorRT FP16 engine:

```bash
cargo run --release -- --trt --fp16 --model <MODEL> --source <SOURCE>
```

Set --device_id to select the specific device. If you have only one GPU and set device_id to 1, the "ort" crate will automatically fall back to the CPU execution provider:

```bash
cargo run --release -- --cuda --device_id 0 --model <MODEL> --source <SOURCE>
```

Set --batch to perform multi-batch-size inference. 

If you are using --trt, you can also set --batch-min and --batch-max to explicitly specify the min/max/opt batch for dynamic batch input. Note that the ONNX model should be exported with dynamic shapes.

```bash
cargo run --release -- --cuda --batch 2 --model <MODEL> --source <SOURCE>
```

Set --height and --width to perform dynamic image size inference (requires a model exported with dynamic shapes):

```bash
cargo run --release -- --cuda --width 480 --height 640 --model <MODEL> --source <SOURCE>
```

Set --profile to check the time consumed in each stage. Note that the model usually requires 1 to 3 dry runs to warm up.

```bash
cargo run --release -- --trt --fp16 --profile --model <MODEL> --source <SOURCE>
```

Results: (yolov8m.onnx, batch=1, 3 times, trt, fp16, RTX 3060Ti)

```text
==> 0
[Model Preprocess]: 12.75788ms
[ORT H2D]: 237.118 us
[ORT Inference]: 507.895469ms
[ORT D2H]: 191.655 us
[Model Inference]: 508.34589ms
[Model Postprocess]: 1.061122ms
==> 1
[Model Preprocess]: 13.658655ms
[ORT H2D]: 209.975 us
[ORT Inference]: 5.12372ms
[ORT D2H]: 182.389 us
[Model Inference]: 5.530022ms
[Model Postprocess]: 1.04851ms
==> 2
[Model Preprocess]: 12.475332ms
[ORT H2D]: 246.127 us
[ORT Inference]: 5.048432ms
[ORT D2H]: 187.117 us
[Model Inference]: 5.493119ms
[Model Postprocess]: 1.040906ms
```

Additional Arguments:

--conf: confidence threshold [default: 0.3]
--iou: iou threshold in NMS [default: 0.45]
--kconf: confidence threshold of keypoint [default: 0.55]
--plot: plot inference result with random RGB color and save

You can check all CLI arguments by running:

```bash
cargo run --release -- --help
```

## Examples

### Classification

Running a dynamic shape ONNX model on CPU with image size --height 224 --width 224.

```bash
cargo run --release -- --model ../assets/weights/yolov8m-cls-dyn.onnx --source ../assets/images/dog.jpg --height 224 --width 224 --plot --profile
```

Example Output:

```text
Summary:
> Task: Classify (Ultralytics 8.0.217)
> EP: Cpu
> Dtype: Float32
> Batch: 1 (Dynamic), Height: 224 (Dynamic), Width: 224 (Dynamic)
> nc: 1000 nk: 0, nm: 0, conf: 0.3, kconf: 0.55, iou: 0.45

[Model Preprocess]: 16.363477ms
[ORT H2D]: 50.722 us
[ORT Inference]: 16.295808ms
[ORT D2H]: 8.37 us
[Model Inference]: 16.367046ms
[Model Postprocess]: 3.527 us
[
    YOLOResult {
        Probs(top5): Some([(208, 0.6950566), (209, 0.13823675), (178, 0.04849795), (215, 0.019029364), (212, 0.016506357)]),
        Bboxes: None,
        Keypoints: None,
        Masks: None,
    },
]
```

### Object Detection

Using CUDA execution provider and dynamic image size:

```bash
cargo run --release -- --cuda --model ../assets/weights/yolov8m-dynamic.onnx --source ../assets/images/bus.jpg --plot --height 640 --width 480
```

### Pose Detection

Using TensorRT execution provider:

```bash
cargo run --release -- --trt --model ../assets/weights/yolov8m-pose.onnx --source ../assets/images/bus.jpg --plot
```

### Instance Segmentation

Using TensorRT execution provider and FP16 model:

```bash
cargo run --release -- --trt --fp16 --model ../assets/weights/yolov8m-seg.onnx --source ../assets/images/0172.jpg --plot
```

## Maintainer

Deepak Rambarki is an AI Engineer with over 2 years of professional experience in developing high-performance machine learning solutions. With a background in Python, C++, and deep learning frameworks like TensorFlow and PyTorch, he focuses on optimizing computer vision pipelines for real-time applications.

Contact: rambarkideepak@gmail.com