#include "roi_conversion.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <cuda_runtime.h>

namespace py = pybind11;

#define checkRuntime(call)  check_runtime(call, #call, __LINE__, __FILE__)

static bool __inline__ check_runtime(cudaError_t e, const char* call, int line, const char *file){
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

static std::vector<uint8_t> load_file(const char* file, size_t expect_size){
    FILE* f = fopen(file, "rb");
    if(f == nullptr){
        printf("Failed to open file: %s\n", file);
        return {};
    } 

    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if(fsize != expect_size){
        printf("Mismatched file size: %d, expected is %d\n", fsize, expect_size);
        fclose(f);
        return {};
    }

    std::vector<uint8_t> host_data(fsize);
    if(fread(host_data.data(), 1, fsize, f) != fsize){
        printf("Failed to read %s bytes from file: %s\n", fsize, file);
        fclose(f);
        return {};
    }
    fclose(f);
    return host_data;
}

struct NV12BlockLinear{
    cudaTextureObject_t luma = 0;     // y
    cudaTextureObject_t chroma = 0;     // uv
    cudaArray_t luma_array;
    cudaArray_t chroma_array;
    unsigned int width = 0, height = 0;
    unsigned int stride = 0;
};

void free_nv12blocklinear(NV12BlockLinear* ptr){
    if(ptr == nullptr) return;

    checkRuntime(cudaDestroyTextureObject(ptr->luma));
    checkRuntime(cudaDestroyTextureObject(ptr->chroma));
    checkRuntime(cudaFreeArray(ptr->luma_array));
    checkRuntime(cudaFreeArray(ptr->chroma_array));
    delete ptr;
}

std::shared_ptr<NV12BlockLinear> load_nv12blocklinear(const char* nv12pl_file, int width, int height){

    auto host_data = load_file(nv12pl_file, width * height * 3 / 2);
    if(host_data.empty()) return nullptr;

    NV12BlockLinear* output = new NV12BlockLinear();
    output->width  = width;
    output->height = height;
    output->stride = width;
    cudaChannelFormatDesc YplaneDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkRuntime(cudaMallocArray(&output->luma_array,   &YplaneDesc, width, height, 0));
    checkRuntime(cudaMemcpy2DToArray(output->luma_array, 0, 0, host_data.data(), width, width, height, cudaMemcpyHostToDevice));

    // One pixel of the uv channel contains 2 bytes
    cudaChannelFormatDesc UVplaneDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
    checkRuntime(cudaMallocArray(&output->chroma_array, &UVplaneDesc, width / 2, height / 2, 0));
    checkRuntime(cudaMemcpy2DToArray(output->chroma_array, 0, 0, host_data.data() + width * height, width, width, height / 2, cudaMemcpyHostToDevice));

    cudaResourceDesc luma_desc = {};
    luma_desc.resType         = cudaResourceTypeArray;
    luma_desc.res.array.array = output->luma_array;

    cudaTextureDesc texture_desc = {};
    texture_desc.filterMode = cudaFilterModePoint;
    texture_desc.readMode   = cudaReadModeElementType;
    checkRuntime(cudaCreateTextureObject(&output->luma, &luma_desc, &texture_desc, NULL));

    cudaResourceDesc chroma_desc = {};
    chroma_desc.resType         = cudaResourceTypeArray;
    chroma_desc.res.array.array = output->chroma_array;
    checkRuntime(cudaCreateTextureObject(&output->chroma, &chroma_desc, &texture_desc, NULL));
    return std::shared_ptr<NV12BlockLinear>(output, free_nv12blocklinear);
}

struct NV12PitchLinear{
    void* luma = nullptr;     // y
    void* chroma = nullptr;     // uv
    unsigned int width = 0, height = 0;
    unsigned int stride = 0;
};

void free_nv12pitchlinear(NV12PitchLinear* ptr){
    if(ptr == nullptr) return;
    checkRuntime(cudaFree(ptr->luma));
    checkRuntime(cudaFree(ptr->chroma));
    delete ptr;
}

std::shared_ptr<NV12PitchLinear> load_nv12pitchlinear(const char* nv12pl_file, int width, int height){

    auto host_data = load_file(nv12pl_file, width * height * 3 / 2);
    if(host_data.empty()) return nullptr;

    NV12PitchLinear* output = new NV12PitchLinear();
    output->width  = width;
    output->height = height;
    output->stride = width;
    checkRuntime(cudaMalloc(&output->luma, width * height));
    checkRuntime(cudaMalloc(&output->chroma, width * height / 2));
    checkRuntime(cudaMemcpy(output->luma, host_data.data(), width * height, cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(output->chroma, host_data.data() + width * height, width * height / 2, cudaMemcpyHostToDevice));
    return std::shared_ptr<NV12PitchLinear>(output, free_nv12pitchlinear);
}

struct YUV422Packed_YUYV{
    void* data = nullptr;     // y
    unsigned int width = 0, height = 0;
    unsigned int stride = 0;
};

void free_yuv422packed_yuyv(YUV422Packed_YUYV* ptr){
    if(ptr == nullptr) return;
    checkRuntime(cudaFree(ptr->data));
    delete ptr;
}

std::shared_ptr<YUV422Packed_YUYV> load_yuv422packed_yuyv(const char* nv12pl_file, int width, int height){

    auto host_data = load_file(nv12pl_file, width * height * 2);
    if(host_data.empty()) return nullptr;

    YUV422Packed_YUYV* output = new YUV422Packed_YUYV();
    output->width  = width;
    output->height = height;
    output->stride = width * 2;
    checkRuntime(cudaMalloc(&output->data, width * height * 2));
    checkRuntime(cudaMemcpy(output->data, host_data.data(), width * height * 2, cudaMemcpyHostToDevice));
    return std::shared_ptr<YUV422Packed_YUYV>(output, free_yuv422packed_yuyv);
}

struct YUVI420Separated{
    void* y = nullptr;     // y
    void* u = nullptr;     // u
    void* v = nullptr;     // v
    unsigned int width = 0, height = 0;
    unsigned int stride = 0;
};

void free_yuvi420separated(YUVI420Separated* ptr){
    if(ptr == nullptr) return;
    checkRuntime(cudaFree(ptr->y));
    checkRuntime(cudaFree(ptr->u));
    checkRuntime(cudaFree(ptr->v));
    delete ptr;
}

std::shared_ptr<YUVI420Separated> load_yuvi420separated(const char* nv12pl_file, int width, int height){

    auto host_data = load_file(nv12pl_file, width * height * 3 / 2);
    if(host_data.empty()) return nullptr;

    YUVI420Separated* output = new YUVI420Separated();
    output->width  = width;
    output->height = height;
    output->stride = width;
    checkRuntime(cudaMalloc(&output->y, width * height));
    checkRuntime(cudaMalloc(&output->u, width * height / 4));
    checkRuntime(cudaMalloc(&output->v, width * height / 4));
    checkRuntime(cudaMemcpy(output->y, host_data.data(), width * height, cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(output->u, host_data.data() + width * height, width * height / 4, cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(output->v, host_data.data() + width * height + width * height / 4, width * height / 4, cudaMemcpyHostToDevice));
    return std::shared_ptr<YUVI420Separated>(output, free_yuvi420separated);
}

struct RGBImage{
    void* data = nullptr; 
    unsigned int width = 0, height = 0;
    unsigned int stride = 0;
};

void free_rgbimage(RGBImage* ptr){
    if(ptr == nullptr) return;
    checkRuntime(cudaFree(ptr->data));
    delete ptr;
}

std::shared_ptr<RGBImage> load_rgbimage(const char* file, int width, int height){

    auto host_data = load_file(file, width * height * 3);
    if(host_data.empty()) return nullptr;
 
    RGBImage* output = new RGBImage();
    output->width  = width;
    output->height = height;
    output->stride = width * 3;
    checkRuntime(cudaMalloc(&output->data, width * height * 3));
    checkRuntime(cudaMemcpy(output->data, host_data.data(), width * height * 3, cudaMemcpyHostToDevice));
    return std::shared_ptr<RGBImage>(output, free_rgbimage);
}

struct RGBAImage{
    void* data = nullptr; 
    unsigned int width = 0, height = 0;
    unsigned int stride = 0;
};

void free_rgbaimage(RGBAImage* ptr){
    if(ptr == nullptr) return;
    checkRuntime(cudaFree(ptr->data));
    delete ptr;
}

std::shared_ptr<RGBAImage> load_rgbaimage(const char* file, int width, int height){

    auto host_data = load_file(file, width * height * 4);
    if(host_data.empty()) return nullptr;
 
    RGBAImage* output = new RGBAImage();
    output->width  = width;
    output->height = height;
    output->stride = width * 4;
    checkRuntime(cudaMalloc(&output->data, width * height * 4));
    checkRuntime(cudaMemcpy(output->data, host_data.data(), width * height * 4, cudaMemcpyHostToDevice));
    return std::shared_ptr<RGBAImage>(output, free_rgbaimage);
}

PYBIND11_MODULE(roiconv, m){

     py::class_<NV12BlockLinear, std::shared_ptr<NV12BlockLinear>>(m, "load_nv12blocklinear")
        .def(py::init([](const char* nv12pl_file, int width, int height){
            return load_nv12blocklinear(nv12pl_file, width, height);
        }))
        .def_property_readonly("width", [](NV12BlockLinear& self){return self.width;})
        .def_property_readonly("height", [](NV12BlockLinear& self){return self.height;})
        .def_property_readonly("stride", [](NV12BlockLinear& self){return self.stride;})
        .def_property_readonly("luma", [](NV12BlockLinear& self){return (uint64_t)self.luma;})
        .def_property_readonly("chroma", [](NV12BlockLinear& self){return (uint64_t)self.chroma;})
        .def("__repr__", [](NV12BlockLinear& self){
            std::stringstream repr;
            repr << "[NV12BlockLinear object]\n";
            repr << "  width=" << self.width << ", stride=" << self.stride << ", height=" << self.height << ", ";
            repr << "  luma=" << self.luma << ", chroma=" << self.chroma;
            return repr.str();
        });
     
     py::class_<NV12PitchLinear, std::shared_ptr<NV12PitchLinear>>(m, "load_nv12pitchlinear")
        .def(py::init([](const char* nv12pl_file, int width, int height){
            return load_nv12pitchlinear(nv12pl_file, width, height);
        }))
        .def_property_readonly("width", [](NV12PitchLinear& self){return self.width;})
        .def_property_readonly("height", [](NV12PitchLinear& self){return self.height;})
        .def_property_readonly("stride", [](NV12PitchLinear& self){return self.stride;})
        .def_property_readonly("luma", [](NV12PitchLinear& self){return (uint64_t)self.luma;})
        .def_property_readonly("chroma", [](NV12PitchLinear& self){return (uint64_t)self.chroma;})
        .def("__repr__", [](NV12PitchLinear& self){
            std::stringstream repr;
            repr << "[NV12PitchLinear object]\n";
            repr << "  width=" << self.width << ", stride=" << self.stride << ", height=" << self.height << ", ";
            repr << "  luma=" << self.luma << ", chroma=" << self.chroma;
            return repr.str();
        });
     
     py::class_<YUV422Packed_YUYV, std::shared_ptr<YUV422Packed_YUYV>>(m, "load_yuv422packed_yuyv")
        .def(py::init([](const char* nv12pl_file, int width, int height){
            return load_yuv422packed_yuyv(nv12pl_file, width, height);
        }))
        .def_property_readonly("width", [](YUV422Packed_YUYV& self){return self.width;})
        .def_property_readonly("height", [](YUV422Packed_YUYV& self){return self.height;})
        .def_property_readonly("stride", [](YUV422Packed_YUYV& self){return self.stride;})
        .def_property_readonly("data", [](YUV422Packed_YUYV& self){return (uint64_t)self.data;})
        .def("__repr__", [](YUV422Packed_YUYV& self){
            std::stringstream repr;
            repr << "[YUV422Packed_YUYV object]\n";
            repr << "  width=" << self.width << ", stride=" << self.stride << ", height=" << self.height << ", ";
            repr << "  data=" << self.data;
            return repr.str();
        });

     py::class_<YUVI420Separated, std::shared_ptr<YUVI420Separated>>(m, "load_yuvi420separated")
        .def(py::init([](const char* nv12pl_file, int width, int height){
            return load_yuvi420separated(nv12pl_file, width, height);
        }))
        .def_property_readonly("width", [](YUVI420Separated& self){return self.width;})
        .def_property_readonly("height", [](YUVI420Separated& self){return self.height;})
        .def_property_readonly("stride", [](YUVI420Separated& self){return self.stride;})
        .def_property_readonly("y", [](YUVI420Separated& self){return (uint64_t)self.y;})
        .def_property_readonly("u", [](YUVI420Separated& self){return (uint64_t)self.u;})
        .def_property_readonly("v", [](YUVI420Separated& self){return (uint64_t)self.v;})
        .def("__repr__", [](YUVI420Separated& self){
            std::stringstream repr;
            repr << "[YUVI420Separated object]\n";
            repr << "  width=" << self.width << ", stride=" << self.stride << ", height=" << self.height << ", ";
            repr << "  y=" << self.y << ",  u=" << self.u << ",  v=" << self.v;
            return repr.str();
        });

     py::class_<RGBAImage, std::shared_ptr<RGBAImage>>(m, "load_rgbaimage")
        .def(py::init([](const char* nv12pl_file, int width, int height){
            return load_rgbaimage(nv12pl_file, width, height);
        }))
        .def_property_readonly("width", [](RGBAImage& self){return self.width;})
        .def_property_readonly("height", [](RGBAImage& self){return self.height;})
        .def_property_readonly("stride", [](RGBAImage& self){return self.stride;})
        .def_property_readonly("data", [](RGBAImage& self){return (uint64_t)self.data;})
        .def("__repr__", [](RGBAImage& self){
            std::stringstream repr;
            repr << "[RGBAImage object]\n";
            repr << "  width=" << self.width << ", stride=" << self.stride << ", height=" << self.height << ", ";
            repr << "  data=" << self.data;
            return repr.str();
        });

     py::class_<RGBImage, std::shared_ptr<RGBImage>>(m, "load_rgbimage")
        .def(py::init([](const char* nv12pl_file, int width, int height){
            return load_rgbimage(nv12pl_file, width, height);
        }))
        .def_property_readonly("width", [](RGBImage& self){return self.width;})
        .def_property_readonly("height", [](RGBImage& self){return self.height;})
        .def_property_readonly("stride", [](RGBImage& self){return self.stride;})
        .def_property_readonly("data", [](RGBImage& self){return (uint64_t)self.data;})
        .def("__repr__", [](RGBImage& self){
            std::stringstream repr;
            repr << "[RGBImage object]\n";
            repr << "  width=" << self.width << ", stride=" << self.stride << ", height=" << self.height << ", ";
            repr << "  data=" << self.data;
            return repr.str();
        });

    py::enum_<roiconv::InputFormat>(m, "InputFormat")
        .value("NV12BlockLinear", roiconv::InputFormat::NV12BlockLinear)
        .value("NV12PitchLinear", roiconv::InputFormat::NV12PitchLinear)
        .value("RGB", roiconv::InputFormat::RGB)
        .value("RGBA", roiconv::InputFormat::RGBA)
        .value("YUV422Packed_YUYV", roiconv::InputFormat::YUV422Packed_YUYV)
        .value("YUVI420Separated", roiconv::InputFormat::YUVI420Separated);

    py::enum_<roiconv::OutputDType>(m, "OutputDType")
        .value("Float16", roiconv::OutputDType::Float16)
        .value("Float32", roiconv::OutputDType::Float32)
        .value("Uint8", roiconv::OutputDType::Uint8);
        // .value("Int32", roiconv::OutputDType::Int32)
        // .value("Int8", roiconv::OutputDType::Int8)
        // .value("Uint32", roiconv::OutputDType::Uint32)

    py::enum_<roiconv::OutputFormat>(m, "OutputFormat")
        .value("CHW16_BGR", roiconv::OutputFormat::CHW16_BGR)
        .value("CHW16_RGB", roiconv::OutputFormat::CHW16_RGB)
        .value("CHW32_BGR", roiconv::OutputFormat::CHW32_BGR)
        .value("CHW32_RGB", roiconv::OutputFormat::CHW32_RGB)
        .value("CHW4_BGR", roiconv::OutputFormat::CHW4_BGR)
        .value("CHW4_RGB", roiconv::OutputFormat::CHW4_RGB)
        .value("CHW_BGR", roiconv::OutputFormat::CHW_BGR)
        .value("CHW_RGB", roiconv::OutputFormat::CHW_RGB)
        .value("Gray", roiconv::OutputFormat::Gray)
        .value("HWC_BGR", roiconv::OutputFormat::HWC_BGR)
        .value("HWC_RGB", roiconv::OutputFormat::HWC_RGB);

    py::enum_<roiconv::Interpolation>(m, "Interpolation")
        .value("Bilinear", roiconv::Interpolation::Bilinear)
        .value("Nearest", roiconv::Interpolation::Nearest);

    py::class_<roiconv::Task>(m, "Task")
        .def(py::init([](int x0, int y0, int x1, int y1, int input_width, int input_stride, int input_height, std::vector<uint64_t> input_planes, int output_width, int output_height, uint64_t output_ptr, std::vector<float> affine_matrix, std::vector<float> alpha, std::vector<float> beta, std::vector<unsigned char> fillcolor){
            roiconv::Task output;
            output.x0 = x0;
            output.y0 = y0;
            output.x1 = x1;
            output.y1 = y1;
            output.input_width   = input_width;
            output.input_stride  = input_stride;
            output.input_height  = input_height;
            output.output_width  = output_width;
            output.output_height = output_height;
            output.output        = (void*)output_ptr;
            if(input_planes.size() != 3) throw py::value_error();
            if(affine_matrix.size() != 6) throw py::value_error();
            if(alpha.size() != 3) throw py::value_error();
            if(beta.size() != 3) throw py::value_error();
            if(fillcolor.size() != 3) throw py::value_error();
            memcpy(output.input_planes, input_planes.data(), sizeof(output.input_planes));
            memcpy(output.affine_matrix, affine_matrix.data(), sizeof(output.affine_matrix));
            memcpy(output.alpha, alpha.data(), sizeof(output.alpha));
            memcpy(output.beta, beta.data(), sizeof(output.beta));
            memcpy(output.fillcolor, fillcolor.data(), sizeof(output.fillcolor));
            return output;
        }))
        .def_property("x0", [](roiconv::Task& self){return self.x0;}, [](roiconv::Task& self, int new_value){self.x0 = new_value;})
        .def_property("y0", [](roiconv::Task& self){return self.y0;}, [](roiconv::Task& self, int new_value){self.y0 = new_value;})
        .def_property("x1", [](roiconv::Task& self){return self.x1;}, [](roiconv::Task& self, int new_value){self.x1 = new_value;})
        .def_property("y1", [](roiconv::Task& self){return self.y1;}, [](roiconv::Task& self, int new_value){self.y1 = new_value;})
        .def_property("input_planes", [](roiconv::Task& self){return std::vector<uint64_t>((uint64_t*)self.input_planes, (uint64_t*)self.input_planes + 3);}, [](roiconv::Task& self, const std::vector<uint64_t>& new_value){if(new_value.size() != 3) throw py::value_error(); memcpy(self.input_planes, new_value.data(), sizeof(self.input_planes));})
        .def_property("input_width", [](roiconv::Task& self){return self.input_width;}, [](roiconv::Task& self, int new_value){self.input_width = new_value;})
        .def_property("input_stride", [](roiconv::Task& self){return self.input_stride;}, [](roiconv::Task& self, int new_value){self.input_stride = new_value;})
        .def_property("input_height", [](roiconv::Task& self){return self.input_height;}, [](roiconv::Task& self, int new_value){self.input_height = new_value;})
        .def_property("output", [](roiconv::Task& self){return self.output;}, [](roiconv::Task& self, uint64_t new_value){self.output = (void*)new_value;})
        .def_property("output_width", [](roiconv::Task& self){return self.output_width;}, [](roiconv::Task& self, int new_value){self.output_width = new_value;})
        .def_property("output_height", [](roiconv::Task& self){return self.output_height;}, [](roiconv::Task& self, int new_value){self.output_height = new_value;})
        .def_property("affine_matrix", [](roiconv::Task& self){return std::vector<float>(self.affine_matrix, self.affine_matrix + 6);}, [](roiconv::Task& self, const std::vector<float>& new_value){if(new_value.size() != 6) throw py::value_error(); memcpy(self.affine_matrix, new_value.data(), sizeof(self.affine_matrix));})
        .def_property("alpha", [](roiconv::Task& self){return std::vector<float>(self.alpha, self.alpha + 3);}, [](roiconv::Task& self, const std::vector<float>& new_value){if(new_value.size() != 3) throw py::value_error(); memcpy(self.alpha, new_value.data(), sizeof(self.alpha));})
        .def_property("beta", [](roiconv::Task& self){return std::vector<float>(self.beta, self.beta + 3);}, [](roiconv::Task& self, const std::vector<float>& new_value){if(new_value.size() != 3) throw py::value_error(); memcpy(self.beta, new_value.data(), sizeof(self.beta));})
        .def_property("fillcolor", [](roiconv::Task& self){return std::vector<unsigned char>(self.fillcolor, self.fillcolor + 3);}, [](roiconv::Task& self, const std::vector<unsigned char>& new_value){if(new_value.size() != 3) throw py::value_error(); memcpy(self.fillcolor, new_value.data(), sizeof(self.fillcolor));})
        .def("resize_affine", [](roiconv::Task& self){self.resize_affine();})
        .def("center_resize_affine", [](roiconv::Task& self){self.center_resize_affine();})
        .def("__repr__", [](roiconv::Task& self){
            std::stringstream repr;
            repr << "[roiconv::Task object]\n";
            repr << "  x0=" << self.x0 << ", y0=" << self.y0 << ", x1=" << self.x1 << ", y1=" << self.y1 << "\n";
            repr << "  input_width=" << self.input_width << ", input_height=" << self.input_height << ", input_stride=" << self.input_stride << "\n";
            repr << "  input_planes=[" << self.input_planes[0] << ", " << self.input_planes[1] << ", " << self.input_planes[2] << "]\n";
            repr << "  output_width=" << self.output_width << ", output_height=" << self.output_height << ", output=" << self.output << "\n";
            repr << "  affine_matrix=[" << self.affine_matrix[0] << ", " << self.affine_matrix[1] << ", " << self.affine_matrix[2] << ", " << self.affine_matrix[3] << ", " << self.affine_matrix[4] << ", " << self.affine_matrix[5] << "]\n";
            repr << "  alpha=[" << self.alpha[0] << ", " << self.alpha[1] << ", " << self.alpha[2] << "]\n";
            repr << "  beta=[" << self.beta[0] << ", " << self.beta[1] << ", " << self.beta[2] << "]\n";
            repr << "  fillcolor=" << (int)self.fillcolor[0] << ", " << (int)self.fillcolor[1] << ", " << (int)self.fillcolor[2];
            return repr.str();
        });

    py::class_<roiconv::ROIConversion, std::shared_ptr<roiconv::ROIConversion>>(m, "ROIConversion")
        .def(py::init([](){
            return roiconv::create();
        }))
        .def("add", [](roiconv::ROIConversion& self, roiconv::Task task){
            return self.add(task);
        })
        .def("run", [](roiconv::ROIConversion& self, roiconv::InputFormat input_format, roiconv::OutputDType output_dtype, roiconv::OutputFormat output_format, roiconv::Interpolation interpolation, uint64_t stream, bool sync, bool clear){
            return self.run(input_format, output_dtype, output_format, interpolation, (void*)stream, sync, clear);
        });
}