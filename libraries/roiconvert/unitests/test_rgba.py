import roiconv
import cv2
import torch

roi = 100, 30, 200, 250
image = cv2.imread("data/ILSVRC2012_val_00023535.JPEG")
cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
# image = cv2.resize(image, dsize=(1920, 1080))
torch_image = torch.from_numpy(image).cuda()
torch_image = torch.cat([torch_image, torch.randn(torch_image.size(0), torch_image.size(1), 1, device="cuda").to(torch.uint8)], dim=-1)
output_tensor = torch.zeros(544, 960, 3).cuda()
print(image.shape)
print(f"Startup")
convtor = roiconv.ROIConversion()
task = roiconv.Task(
    *roi,
    image.shape[1], image.shape[1] * 4, image.shape[0],
    [torch_image.data_ptr(), 0, 0], 
    960, 544,
    output_tensor.data_ptr(),
    [1, 0, 0, 0, 1, 0], 
    [1, 1, 1], 
    [0, 0, 0], 
    [128, 128, 128]
)
task.center_resize_affine()
print(task)
convtor.add(task)

ok = convtor.run(roiconv.InputFormat.RGBA, roiconv.OutputDType.Float32, roiconv.OutputFormat.HWC_RGB, roiconv.Interpolation.Nearest, 0, True, True)
print(f"OK = {ok}")
ok = convtor.run(roiconv.InputFormat.RGBA, roiconv.OutputDType.Float32, roiconv.OutputFormat.HWC_RGB, roiconv.Interpolation.Nearest, 0, True, True)
print(f"OK = {ok}")

cv2.imwrite("output.png", output_tensor.to(torch.uint8).cpu().data.numpy())
cv2.imwrite("image.jpg", image)