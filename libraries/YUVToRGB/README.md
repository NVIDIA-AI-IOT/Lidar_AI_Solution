# YUVToRGB
Implemention of YUV -> RGB batch convertion with single cuda kernel.
![title](/assets/yuvtorgb.png)

## 1. Supported Feature
- Input can be YUV_NV12 Block Linear or YUV_NV12 Pitch Linear or YUV_YUYV Packed
- Output can be RGB/BGR NCHW or RGB/BGR NHWC for GPU and RGB/BGR NCHW16 for DLA
- Conversion formula R/G/B_output = (R/G/B - offset_R/G/B) * scale_R/G/B
- Rescale interpolation support Nearest and Bilinear mode
- Verifed to keep exactly the same output as OpenCV
- Async API to run on specific CUDA stream
- Command line:
```bash
$ ./yuvtorgb --help
Usage: ./yuvtorgb --input=3840x2160x1/BL --output=1280x720/uint8/NCHW_RGB --interp=nearest --save=tensor.binary --perf

parameters:
    --input:  Set input size and format, Syntax format is: [width]x[height]x[batch]/[format]
              format can be 'BL' or 'PL' or 'YUYV' 
    --output: Set output size and layout, Syntax format is: [width]x[height]/[dtype]/[layout]
              dtype can be 'uint8', 'float16' or 'float32'
              layout can be one of the following: NCHW_RGB NCHW_BGR NHWC_RGB NHWC_BGR for GPU, NCHW16_RGB NCHW16_BGR for DLA
    --interp: Set rescale mode. Here the choice 'nearest' or 'bilinear', default is nearest
    --save:   Set the path of the output. default does not save the output
    --perf:   Launch performance test with 1000x warmup and 1000x iteration
```

## 2. Performance
### 2.1 Performance Table
<table>
<thead>
  <tr>
    <th>Input</th>
    <th colspan="6">Block Linear YUV</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Foumal</td>
    <td colspan="6">R/G/B_Output = (R/G/B - offset_R/G/B) * scale_R/G/B</td>
  </tr>
  <tr>
    <td>Environment</td>
    <td colspan="6">Orin-DOS6.0.3.0(30270381) / CUDA11.4.15 / BATCH=1 / CPU@2010MHz / GPU@1275MHz / EMC@3199MHz</td>
  </tr>
  <tr>
    <td>Output(RGB)</td>
    <td colspan="2">NHWC (us)</td>
    <td colspan="2">NCHW (us)</td>
    <td colspan="2">NCHW16 (us)</td>
  </tr>
  <tr>
    <td>Input-Output</td>
    <td>3840x2160-1920x1080</td>
    <td>1920x1080-1080x720</td>
    <td>3840x2160-1920x1080</td>
    <td>1920x1080-1080x720</td>
    <td>3840x2160-1920x1080</td>
    <td>1920x1080-1080x720</td>
  </tr>
  <tr>
    <td>FP32/Nearest</td>
    <td>314.69</td>
    <td>124.63</td>
    <td>247.62</td>
    <td>89.54</td>
    <td>1221.38</td>
    <td>437.55</td>
  </tr>
  <tr>
    <td>FP16/Nearest</td>
    <td>287.11</td>
    <td>112.19</td>
    <td>168.13</td>
    <td>64.26</td>
    <td>1037.24</td>
    <td>353.84</td>
  </tr>
  <tr>
    <td>UINT8/Nearest</td>
    <td>272.24</td>
    <td>105.85</td>
    <td>136.06</td>
    <td>54.21</td>
    <td>632.93</td>
    <td>202.67</td>
  </tr>
  <tr>
    <td>FP32/Bilinear</td>
    <td>549.77</td>
    <td>202.69</td>
    <td>472.32</td>
    <td>183.85</td>
    <td>1449.23</td>
    <td>508.97</td>
  </tr>
  <tr>
    <td>FP16/Bilinear</td>
    <td>518.23</td>
    <td>194.04</td>
    <td>371.32</td>
    <td>145.82</td>
    <td>1145.20</td>
    <td>386.06</td>
  </tr>
  <tr>
    <td>UINT8/Bilinear</td>
    <td>503.39</td>
    <td>189.02</td>
    <td>435.90</td>
    <td>169.14</td>
    <td>756.36</td>
    <td>264.71</td>
  </tr>
</tbody>
</table>

### 2.2 How to Benchmark
- step1: `make yuvtorgb` to generate program.
- step2: `./yuvtorgb --input=3840x2160x1/BL --output=1280x720/uint8/NCHW_RGB --interp=nearest --perf`
  - Simple performance testing in a specific configuration
```
$./yuvtorgb --input=3840x2160x1/BL --output=1280x720/uint8/NCHW_RGB --interp=nearest --perf
[Nearest] 3840x2160x1/NV12BlockLinear to 1280x720/Uint8/NCHW_RGB performance: 30.32 us
```

### 2.3 How to Verify the Accuracy
- make check : Generate all the binary files and call the python script for error checking
- Verification for 2 aspects below:
  - Color space conversion
    - The same formula is used, so the exact same result is obtained
  - Rescale interpolation mode
    - Nearest: Verify at multiple resolutions and get the exact same results
    - Bilinear: When the scaling factor used is a rational number, exactly the same result can be obtained. Otherwise, the deviation is no more than 1 pixel.
```
$ pip install numpy opencv-python
$ make compare
Compile depends CUDA src/yuv_to_rgb_kernel.cu
Compile depends C++ src/main.cpp
Compile depends C++ src/yuv_to_rgb.cpp
Compile CXX src/yuv_to_rgb.cpp
Compile CXX src/main.cpp
Compile CUDA src/yuv_to_rgb_kernel.cu
Link yuvtorgb
rm -rf workspace/*.bin workspace/*.png
python ./compare-with-opencv.py
nearest: 3840x2160x1/BL to 1920x1280/uint8/NCHW16_BGR, diff sum = 0, avg = 0.0, max = 0, std = 0.0
nearest: 3840x2160x1/BL to 1920x1280/uint8/NCHW16_RGB, diff sum = 0, avg = 0.0, max = 0, std = 0.0
nearest: 3840x2160x1/BL to 1920x1280/uint8/NCHW_BGR, diff sum = 0, avg = 0.0, max = 0, std = 0.0
```

## 3. User Integration
- Only need to include *yuvtorgb_library/yuv_to_rgb_kernel.cu* and *yuvtorgb_library/yuv_to_rgb_kernel.hpp* for integration.
- Pure C-style interface has been provided.
- To make the library: `nvcc -Xcompiler "-fPIC" -shared -O3 yuvtorgb_library/yuv_to_rgb_kernel.cu -o libyuvtorgb_kernel.so -lcudart`

## 4. Summary
- For BL8800 (video decoder default).
  1. UVplane uses cudaCreateChannelDesc(8, 8, 0, 0) when cudaMallocArray
  2. At this moment each pixel takes up 2bytes
  3. When cudaMallocArray, the size of UVplane is: image_width / 2, image_height / 2. Because 2bytes per pixel
  4. cudaMemcpy2DToArrayAsync, when userdata copy to cudaArray, width = image_width, height = image_height / 2. When copying, the width is byte unit.
  5. For the cuda kernel, reading uses `uv = tex2D<uchar2>(chroma, x/2, y/2)`;
- For BL8000.
  1. UVplane uses cudaCreateChannelDesc(8, 0, 0, 0) when cudaMallocArray
  2. At this moment each pixel occupies 1bytes
  3. For cudaMallocArray, the size of UVplane is: image_width, image_height / 2. Because 1bytes per pixel
  4. cudaMemcpy2DToArrayAsync, when userdata copy to cudaArray, width = image_width, height = image_height / 2. When copying, the width is byte unit.
  5. For the cuda kernel, reading uses `u = tex2D<uchar>(chroma, int(x/2)*2+0, y/2); v = tex2D<uchar>(chroma, int(x/2)*2+1, y/2)`;
