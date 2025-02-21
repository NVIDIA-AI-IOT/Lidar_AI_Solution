import roiconv
import torch
import cv2

nv12 = roiconv.load_nv12blocklinear("data/nv12_3840x2160.yuv", 3840, 2160)
output_tensor = torch.zeros(256, 256, 3).cuda()

convtor = roiconv.ROIConversion()
task = roiconv.Task(
    0, 0, nv12.width, nv12.height,
    nv12.width, nv12.stride, nv12.height,
    [nv12.luma, nv12.chroma, 0], 
    256, 256,
    output_tensor.data_ptr(),
    [1, 0, 0, 0, 1, 0], 
    [1, 1, 1], 
    [0, 0, 0], 
    [128, 128, 128]
)
task.center_resize_affine()
print(task)
convtor.add(task)

ok = convtor.run(roiconv.InputFormat.NV12BlockLinear, roiconv.OutputDType.Float32, roiconv.OutputFormat.HWC_BGR, roiconv.Interpolation.Nearest, 0, True, True)
cv2.imwrite("output.png", output_tensor.to(torch.uint8).cpu().data.numpy())