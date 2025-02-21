import roiconv
import torch
import cv2

images = []
convtor = roiconv.ROIConversion()
for i in range(16):
    nv12 = roiconv.load_nv12blocklinear("data/1920x1080-nv12.yuv", 1920, 1080)
    output_tensor = torch.zeros(3, 544, 960).cuda()
    images.append([nv12, output_tensor])

    task = roiconv.Task(
        0, 0, nv12.width, nv12.height,
        nv12.width, nv12.stride, nv12.height,
        [nv12.luma, nv12.chroma, 0], 
        960, 544,
        output_tensor.data_ptr(),
        [1, 0, 0, 0, 1, 0], 
        [1, 1, 1], 
        [0, 0, 0], 
        [128, 128, 128]
    )
    task.resize_affine()
    convtor.add(task)

import time

for i in range(100):
    ok = convtor.run(roiconv.InputFormat.NV12BlockLinear, roiconv.OutputDType.Float32, roiconv.OutputFormat.CHW_RGB, roiconv.Interpolation.Nearest, 0, True, False)

for i in range(100):
    now = time.time()
    ok = convtor.run(roiconv.InputFormat.NV12BlockLinear, roiconv.OutputDType.Float32, roiconv.OutputFormat.CHW_RGB, roiconv.Interpolation.Nearest, 0, True, False)
    gap = time.time() - now
    print(f"{gap * 1000:.3f} ms")

cv2.imwrite("output.png", output_tensor.permute(1, 2, 0).to(torch.uint8).cpu().data.numpy())
