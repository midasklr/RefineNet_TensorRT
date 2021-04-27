# RefineNet_TensorRT

## update20210427
recommend DDRNet whice achieve 334 FPS with TensorRT-FP16
https://github.com/midasklr/DDRNet.TensorRT

TensorRT for RefineNet Segmentation.

This is TensorRT C++ project for [RefineNet](https://github.com/midasklr/RefineNet). See details for how to train refinenet for semantic segmentation and export onnx model.

这里是RefineNet语义分割模型的TensorRT工程，参考之前的[RefineNet](https://github.com/midasklr/RefineNet)，这里我使用Helen人脸分割数据集训练，对原始的RefineNet模型进行了一些修改，修改了部分3*3卷积用于轻量化模型，修改了Upsample为转置卷积。

## environment

Ubuntu1604

Pytorch 0.41

TensorRT5.1.5

OpenCV3.4.8

## Build

1. configure your TensorRT path in CMakeLists.txt

2. make:

   ```
   mkdir build && cd build
   cmake ..
   make -j8
   ```

3. 

   ```
   ./RefineNet s float16(float32) refinenet.engine ../vid/demo.mp4 ../refinenet.onnx
   ```

4. serialize the engine from onnx model:

   ```
   ./RefineNet s float16(float32) refinenet.engine ../vid/face.mp4 ../refinenet.onnx
   ```

5.  deserialize the engine and infer:

   ```
   ./RefineNet infer float16 refinenet.engine ../vid/face.mp4
   ```

   <img src="./image/d94be52120f2aa2cfbd7c12f10817b04.jpeg" style="zoom:25%;" />

   <img src="./image/Screenshot from 2020-08-16 13-39-14.png" style="zoom: 50%;" />

   

## Performance

| Model   | FPS  |
| ------- | ---- |
| Pytorch | 5    |
| FP32    | 27   |
| FP16    | 33   |

