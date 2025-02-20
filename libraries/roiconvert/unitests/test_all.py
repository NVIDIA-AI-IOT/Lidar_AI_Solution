import roiconv
import torch
import numpy as np
import cv2

input_formats = [
    roiconv.InputFormat.NV12BlockLinear,
    roiconv.InputFormat.NV12PitchLinear,
    roiconv.InputFormat.YUV422Packed_YUYV,
    roiconv.InputFormat.YUVI420Separated,
    roiconv.InputFormat.RGBA,
    roiconv.InputFormat.RGB
]

interpolations = [
    roiconv.Interpolation.Bilinear,
    roiconv.Interpolation.Nearest
]

output_dtypes = [
    roiconv.OutputDType.Uint8,
    roiconv.OutputDType.Float32,
    roiconv.OutputDType.Float16,
    # roiconv.OutputDType.Int8,
    # roiconv.OutputDType.Int32,
    # roiconv.OutputDType.Uint32
]

output_formats = [
    roiconv.OutputFormat.CHW_RGB,
    roiconv.OutputFormat.CHW_BGR,
    roiconv.OutputFormat.HWC_RGB,
    roiconv.OutputFormat.HWC_BGR,
    roiconv.OutputFormat.CHW16_RGB,
    roiconv.OutputFormat.CHW16_BGR,
    roiconv.OutputFormat.CHW32_RGB,
    roiconv.OutputFormat.CHW32_BGR,
    roiconv.OutputFormat.CHW4_RGB,
    roiconv.OutputFormat.CHW4_BGR,
    roiconv.OutputFormat.Gray
]

def fill_by_input_format(task:roiconv.Task, input_format:roiconv.InputFormat, vars:dict):
    if input_format == roiconv.InputFormat.NV12BlockLinear:
        nv12 = roiconv.load_nv12blocklinear("data/nv12_3840x2160.yuv", 3840, 2160)
        vars["input_format"] = {"nv12": nv12}
        task.input_width = nv12.width
        task.input_height = nv12.height
        task.input_stride = nv12.stride
        task.input_planes = [nv12.luma, nv12.chroma, 0]
    elif input_format == roiconv.InputFormat.NV12PitchLinear:
        nv12 = roiconv.load_nv12pitchlinear("data/nv12_3840x2160.yuv", 3840, 2160)
        vars["input_format"] = {"nv12": nv12}
        task.input_width = nv12.width
        task.input_height = nv12.height
        task.input_stride = nv12.stride
        task.input_planes = [nv12.luma, nv12.chroma, 0]
    elif input_format == roiconv.InputFormat.YUV422Packed_YUYV:
        yuyv = roiconv.load_yuv422packed_yuyv("data/yuyv_3840x2160_yuyv422.yuv", 3840, 2160)
        vars["input_format"] = {"yuyv": yuyv}
        task.input_width = yuyv.width
        task.input_height = yuyv.height
        task.input_stride = yuyv.stride
        task.input_planes = [yuyv.data, 0, 0]
    elif input_format == roiconv.InputFormat.YUVI420Separated:
        i420 = roiconv.load_yuvi420separated("data/1920x1080-i420.yuv", 1920, 1080)
        vars["input_format"] = {"i420": i420}
        task.input_width = i420.width
        task.input_height = i420.height
        task.input_stride = i420.stride
        task.input_planes = [i420.y, i420.u, i420.v]
    elif input_format == roiconv.InputFormat.RGB:
        rgb = roiconv.load_rgbimage("data/375x500.rgb", 375, 500)
        vars["input_format"] = {"rgb": rgb}
        task.input_width = rgb.width
        task.input_height = rgb.height
        task.input_stride = rgb.stride
        task.input_planes = [rgb.data, 0, 0]
    elif input_format == roiconv.InputFormat.RGBA:
        rgba = roiconv.load_rgbaimage("data/375x500.rgba", 375, 500)
        vars["input_format"] = {"rgba": rgba}
        task.input_width = rgba.width
        task.input_height = rgba.height
        task.input_stride = rgba.stride
        task.input_planes = [rgba.data, 0, 0]
    else:
        raise NotImplementedError(f"Unknow input_format: {input_format}")

def torch_dtype(output_dtype):
    mapping = {
        roiconv.OutputDType.Uint8   : torch.uint8,
        roiconv.OutputDType.Float32 : torch.float32,
        roiconv.OutputDType.Float16 : torch.float16,
        # roiconv.OutputDType.Int8    : torch.int8,
        # roiconv.OutputDType.Int32   : torch.int32,
        # roiconv.OutputDType.Uint32  : torch.uint32
    }
    if output_dtype not in mapping:
        raise NotImplementedError(f"Unknow output dtype: {output_dtype}")
    return mapping[output_dtype]

def make_output_tensor(output_dtype, output_format, width, height):
    shapes = [height, width]
    if output_format in [roiconv.OutputFormat.HWC_RGB, roiconv.OutputFormat.HWC_BGR]:
        shapes = shapes + [3]
    elif output_format in [roiconv.OutputFormat.CHW_RGB, roiconv.OutputFormat.CHW_BGR]:
        shapes = [3] + shapes
    elif output_format in [roiconv.OutputFormat.CHW16_RGB, roiconv.OutputFormat.CHW16_BGR]:
        shapes = [16] + shapes
    elif output_format in [roiconv.OutputFormat.CHW32_RGB, roiconv.OutputFormat.CHW32_BGR]:
        shapes = [32] + shapes
    elif output_format in [roiconv.OutputFormat.CHW4_RGB, roiconv.OutputFormat.CHW4_BGR]:
        shapes = [4] + shapes
    elif output_format == roiconv.OutputFormat.Gray:
        shapes = shapes + [1]
    else:
        raise NotImplementedError(f"Unknow output format: {output_format}")

    return (torch.rand(*shapes, device="cuda") * 255).to(torch_dtype(output_dtype))

def store_output(i, input_format, output_dtype, output_format, interpolation, tensor):
    if tensor.dtype != torch.int8:
        tensor = tensor.clamp(0, 255)
    tensor = tensor.cpu().data.numpy().astype(np.uint8)
    if output_format in [roiconv.OutputFormat.HWC_RGB, roiconv.OutputFormat.HWC_BGR]:
        image = tensor
    elif output_format in [roiconv.OutputFormat.CHW_RGB, roiconv.OutputFormat.CHW_BGR]:
        image = tensor.transpose(1, 2, 0)
    elif output_format in [roiconv.OutputFormat.CHW16_RGB, roiconv.OutputFormat.CHW16_BGR]:
        image = tensor.reshape(tensor.shape[1], tensor.shape[2], -1)[..., :3]
    elif output_format in [roiconv.OutputFormat.CHW32_RGB, roiconv.OutputFormat.CHW32_BGR]:
        image = tensor.reshape(tensor.shape[1], tensor.shape[2], -1)[..., :3]
    elif output_format in [roiconv.OutputFormat.CHW4_RGB, roiconv.OutputFormat.CHW4_BGR]:
        image = tensor.reshape(tensor.shape[1], tensor.shape[2], -1)[..., :3]
    elif output_format == roiconv.OutputFormat.Gray:
        image = tensor.squeeze(-1)

    name = f"outputs/{i:03d}_{input_format.name}_{output_dtype.name}_{output_format.name}_{interpolation.name}.jpg"
    print(name, image.shape, tensor.shape)
    cv2.imwrite(name, image)

convtor = roiconv.ROIConversion()
task = roiconv.Task(
    15, 31, 256, 256,
    0, 0, 0,
    [0, 0, 0], 
    256, 256,
    0,
    [1, 0, 0, 0, 1, 0], 
    [1, 1, 1], 
    [0, 0, 0], 
    [128, 128, 128]
)

vars = dict()
i = 0
for input_format in input_formats:
    fill_by_input_format(task, input_format, vars)

    for interpolation in [roiconv.Interpolation.Bilinear]:
        for output_dtype in output_dtypes:
            for output_format in output_formats:
                output_tensor = make_output_tensor(output_dtype, output_format, 512, 512)
                task.output_width = 512
                task.output_height = 512
                task.x0 = 0
                task.y0 = 0
                task.x1 = task.input_width
                task.y1 = task.input_height
                task.output = output_tensor.data_ptr()
                task.center_resize_affine()
                convtor.add(task)
                print(f"Run: {input_format}, {interpolation}, {output_dtype}, {output_format}, output: {output_tensor.shape}, {output_tensor.dtype}, min: {output_tensor.cpu().data.numpy().min()}, max: {output_tensor.cpu().data.numpy().max()}")
                assert convtor.run(input_format, output_dtype, output_format, interpolation, 0, True, True), f"Failed to run convertor"
                i += 1
                store_output(i, input_format, output_dtype, output_format, interpolation, output_tensor)

print("Done.")