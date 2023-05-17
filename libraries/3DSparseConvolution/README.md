# 3D Sparse Convolution Network
A tiny inference engine for [3d sparse convolutional networks](https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/backbones/scn.py) using int8/fp16.
![title](/assets/3dsparse_conv.png)

## Model && Data
This demo uses lidar data from [nuScenes Dataset](https://www.nuscenes.org/).
Onnx model can be converted from checkpoint and config below using given script.
|  Dataset  |  Checkpoint  | Config |
| --------- | ------------ | ------ |
|  nuScenes | [epoch_20.pth](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/EhgzjwV2EghOnHFKyRgSadoBr2kUo7yPu52N-I3dG3c5dA?e=a9MdhX) | [nusc_centerpoint_voxelnet_0075voxel_fix_bn_z](https://github.com/tianweiy/CenterPoint/blob/master/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py) |

## Accuracy on nuScenes Validation
|         **Model**        |  **3D Inference** | **Precision** | **mAP** | **NDS** |                   **Description**                  |
|:------------------------:|:-----------------:|:-------------:|:-------:|:-------:|:--------------------------------------------------:|
| centerpoint.scn.PTQ.onnx |     libspconv.so     |      INT8     | 59.15  |  66.45  | PTQ Model, spconv.so INT8 Inference                |
| centerpoint.scn.PTQ.pth | PyTorch FakeQuant |      INT8     |   59.08  |   66.2  | PTQ Model, PyTorch FP16 Inference + FakeQuant-INT8 |
|     [scn.nuscenes.pth](https://github.com/tianweiy/CenterPoint/blob/master/configs/nusc/README.md)     |      PyTorch      |      FP16     |   59.6  |   66.8  | From CenterPoint official, Validation              |
|     centerpoint.scn.onnx    |     libspconv.so     |      FP16     |   59.5  |  66.71  | From CenterPoint official & Inference by spconv.so |

## Memory Usage
|     **Model**     | **Precision** | **Memory** |
|:-----------------:|:-------------:|:----------:|
| centerpoint.scn.PTQ.onnx |      FP16     |    422MB   |
| centerpoint.scn.PTQ.onnx |      INT8     |    426MB   |

## Export ONNX
1. Download and configure the CenterPoint environment from https://github.com/tianweiy/CenterPoint
2. Export SCN ONNX
```
$ cp -r tool/centerpoint-export path/to/CenterPoint
$ cd path/to/CenterPoint
$ python centerpoint-export/export-scn.py --ckpt=epoch_20.pth --save-onnx=scn.nuscenes.onnx
$ cp scn.nuscenes.onnx path/to/3DSparseConvolution/workspace/
```

3. ## Compile && Run
- Build and run test
```
$ sudo apt-get install libprotobuf-dev
$ cd path/to/3DSparseConvolution
->>>>>> modify main.cpp:80 to scn.nuscenes.onnx
$ make fp16 -j
🙌 Output.shape: 1 x 256 x 180 x 180
[PASSED 🤗], libspconv version is 1.0.0
To verify the results, you can execute the following command.
Verify Result:
  python tool/compare.py workspace/centerpoint/out_dense.torch.fp16.tensor workspace/centerpoint/output.zyx.dense --detail
[PASSED].
```

- Verify output
```
$ python tool/compare.py workspace/centerpoint/out_dense.torch.fp16.tensor workspace/centerpoint/output.zyx.dense --detail
================ Compare Information =================
 CPP     Tensor: 1 x 256 x 180 x 180, float16 : workspace/centerpoint/out_dense.torch.fp16.tensor
 PyTorch Tensor: 1 x 256 x 180 x 180, float16 : workspace/centerpoint/output.zyx.dense
[absdiff]: max:0.19891357421875, sum:1438.463379, std:0.001725, mean:0.000173
CPP:   absmax:3.066406, min:0.000000, std:0.034445, mean:0.003252
Torch: absmax:3.054688, min:0.000000, std:0.034600, mean:0.003279
[absdiff > m75% --- 0.149185]: 0.000 %, 2
[absdiff > m50% --- 0.099457]: 0.000 %, 17
[absdiff > m25% --- 0.049728]: 0.010 %, 846
[absdiff > 0]: 2.140 %, 177539
[absdiff = 0]: 97.860 %, 8116861
[cosine]: 99.876 %
======================================================
```

## For Python
```
$ make pyscn -j
Use Python Include: /usr/include/python3.8
Use Python SO Name: python3.8
Use Python Library: /usr/lib
Compile CXX src/pyscn.cpp
Link tool/pyscn.so
You can run "python tool/pytest.py" to test

$ python tool/pytest.py
[PASSED 🤗].
To verify result:
  python tool/compare.py workspace/centerpoint/out_dense.py.fp16.tensor workspace/centerpoint/out_dense.torch.fp16.tensor --detail
```

## Performance on ORIN
- Summary performance using 6019 data from nuscenes
![](workspace/perf.png)

## Note
- The current version supports compute arch are required sm_80, sm_86, and sm_87..
- Supported operators:
  - SparseConvolution, Add, Relu, Add&Relu, ScatterDense, Reshape and ScatterDense&Transpose.
- Supported SparseConvolution:
  - SpatiallySparseConvolution and SubmanifoldSparseConvolution.
- Supported properties of SparseConvolution:
  - activation, kernel_size, dilation, stride, padding, rulebook, subm, output_bound, precision and output_precision.