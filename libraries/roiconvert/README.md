# ROI Conversion
ROIs to continuous tensor conversion. This is a library for implementing conversions from any number of ROIs to a continuous output tensor.

## 1. Supported Feature
- Input can be NV12 Block Linear or NV12 Pitch Linear or YUV_YUYV Packed or I420 Separated or RGBA or RGB.
- Output color format can be Gray/BGR/GRB.
- Output network order can be Gray/HWC/CHW/CHW4/CHW16/CHW32.
- Conversion formula R/G/B_output = (R/G/B - offset_R/G/B) * scale_R/G/B
- Rescale interpolation support Nearest and Bilinear mode
- Verifed to keep exactly the same output as OpenCV
- Async API to run on specific CUDA stream

## 2. Unitests
### 2.1 Build
```bash
$ export CUDA_VER=12.6 && make
```

### 2.2 Test Cases
```bash
$ mkdir outputs
$ python3 unitests/test_nv12.py     #Input nv12
$ python3 unitests/test_rgba.py     #Input rgba
$ python3  unitests/test_all.py     #Input all supported formats.
```

### 2.3 Performance

```bash
$ python3  unitests/test_perf.py
```
Performance Table
<table>
<thead>
  <tr>
    <th>Input</th>
    <th colspan="6">NV12 Block Linear</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Foumal</td>
    <td colspan="6">R/G/B_Output = (R/G/B - offset_R/G/B) * scale_R/G/B</td>
  </tr>
  <tr>
    <td>Environment</td>
    <td colspan="6">RTX 6000 Ada / CUDA12.6 / BATCHSIZE=16</td>
  </tr>
  <tr>
    <td>Output(RGB)</td>
    <td>NCHW</td>
  </tr>
  <tr>
    <td>Input-Output</td>
    <td>1920x1080-960x544</td>
  </tr>
  <tr>
    <td>FP32/Nearest</td>
    <td>0.20998ms</td>
  </tr>
</tbody>
</table>

### 2.4 How to Verify the Accuracy
Reszie the RGBA source to a specific size by roi_conversion and opencv respectively. Then compare the two results.
```
$ pip install numpy opencv-python
$ python3 unitests/test_resize.py
(500, 375, 3)
Startup
[roiconv::Task object]
  x0=11, y0=31, x1=75, y1=95
  input_width=375, input_height=500, input_stride=1500
  input_planes=[0x7cfceae89600, 0, 0]
  output_width=256, output_height=256, output=0x7cfcd3a00000
  affine_matrix=[4, 0, 0, 0, 4, 0]
  alpha=[1, 1, 1]
  beta=[0, 0, 0]
  fillcolor=128, 128, 128
OK = True
0.0 0.0 0.0
```

## 3. User Integration
- Only need to include *roi_conversion/roi_conversion.cu* and *roi_conversion/roi_conversion.hpp* for integration.
- Pure C-style interface has been provided.
- To make the library: `nvcc -Xcompiler "-fPIC" -shared -O3 roi_conversion/roi_conversion.cu -o libroiconvert_kernel.so -lcudart`
