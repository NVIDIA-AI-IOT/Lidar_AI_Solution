#ifndef __ROI_CONVERSION_HPP__
#define __ROI_CONVERSION_HPP__

#include <memory>

namespace roiconv{

enum class InputFormat : unsigned int{
    NoneEnum          = 0,
    NV12BlockLinear   = 1,   // Y, UV   stride = width
    NV12PitchLinear   = 2,   // Y, UV   stride = width
    YUV422Packed_YUYV = 3,   // YUV     stride = width * 2
    YUVI420Separated  = 4,   // Y, U, V stride = width
    RGBA              = 5,   // stride = width * 4
    RGB               = 6    // stride = width * 3
};

enum class Interpolation : unsigned int{
    NoneEnum    = 0,
    Nearest     = 1,
    Bilinear    = 2
};

enum class OutputDType : unsigned int{
    NoneEnum    = 0,
    Uint8       = 1,
    Float32     = 2,
    Float16     = 3
    // Int8        = 4,
    // Int32       = 5,
    // Uint32      = 6
};

enum class OutputFormat : unsigned int{
    NoneEnum   = 0,
    CHW_RGB   = 1,
    CHW_BGR   = 2,
    HWC_RGB   = 3,
    HWC_BGR   = 4,
    CHW16_RGB = 5,  // c = (c + 15) / 16 * 16 if c % 16 != 0 else c
    CHW16_BGR = 6,
    CHW32_RGB = 7,  // c = (c + 31) / 32 * 32 if c % 32 != 0 else c
    CHW32_BGR = 8,
    CHW4_RGB  = 9,  // c = (c + 3) / 4 * 4 if c % 4 != 0 else c
    CHW4_BGR  = 10,
    Gray      = 11
};

struct Task{
    int x0, y0, x1, y1;  // source coordinates in pixels.
    const void *input_planes[3];
    int input_width, input_height;
    int input_stride;

    void* output;
    int output_width, output_height;

    float affine_matrix[6];
    float alpha[3], beta[3];
    unsigned char fillcolor[3];

    void resize_affine();
    void center_resize_affine();
};

class ROIConversion{
public:
    virtual void add(const Task& task) = 0;
    virtual bool run(InputFormat input_format, OutputDType output_dtype, OutputFormat output_format, Interpolation interpolation, void* stream, bool sync=false, bool clear=true) = 0;
};

// image = load_image(width, height)
// roi = image(x0, y0, x1, y1)
// roi = affine(roi, matrix, fill=fillcolor, dst_size=(width, height), interp=interpolation)
// roi = roi * alpha + beta
// store(roi, dest, format=dest_format, dtype=dest_dtype)
std::shared_ptr<ROIConversion> create();

}; // namespace roiconv

#endif // __ROI_CONVERSION_HPP__