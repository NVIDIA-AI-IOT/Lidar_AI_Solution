#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <vector>
#include <string>
#include <assert.h>
#include <cuda_fp16.h>

using namespace nvinfer1;

template<typename T>
static void __global__ layernorm_kernel(const T* x, const T* weight, const T* bias, T* y, int N, int C, float epsilon);

template<>
void __global__ layernorm_kernel<float>(const float* x, const float* weight, const float* bias, float* y, int N, int C, float epsilon){
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= N) return;

    // x: N, C
    // y: N, C
    // weight: C
    // bias:   C
    const float* px = x + idx * C;
    float*       py = y + idx * C;

    // reduce sum
    float sq = 0.0f;
    float s  = 0.0f;
    float diver = 1.0f / C;
    for(int ic = threadIdx.x; ic < C; ic += warpSize){
        float x = px[ic];
        s += x;
        sq = fmaf(x, x * diver, sq);
    }

    for (int mask = 16; mask > 0; mask /= 2)
        s += __shfl_xor_sync(0xffffffff, s, mask);

    for (int mask = 16; mask > 0; mask /= 2)
        sq += __shfl_xor_sync(0xffffffff, sq, mask);

    float mean = s / C;
    float rstd = rsqrtf(sq - mean * mean + epsilon);
    for(int ic = threadIdx.x; ic < C; ic += warpSize) 
        py[ic] = (px[ic] - mean) * weight[ic] * rstd + bias[ic];
}

template<>
void __global__ layernorm_kernel<half>(const half* x, const half* weight, const half* bias, half* y, int N, int C, float epsilon){
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= N) return;

    // x: N, C
    // y: N, C
    // weight: C
    // bias:   C
    const half* px = x + idx * C;
          half* py = y + idx * C;

    // reduce sum
    float sq = 0.0f;
    float s  = 0.0f;
    float diver = 1.0f / C;
    for(int ic = threadIdx.x; ic < C; ic += warpSize){
        float x = __half2float(px[ic]);
        s += x;
        sq = fmaf(x, x * diver, sq);
    }

    for (int mask = 16; mask > 0; mask /= 2)
        s += __shfl_xor_sync(0xffffffff, s, mask);

    for (int mask = 16; mask > 0; mask /= 2)
        sq += __shfl_xor_sync(0xffffffff, sq, mask);

    float mean = s / C;
    float rstd = rsqrtf(sq - mean * mean + epsilon);
    for(int ic = threadIdx.x; ic < C; ic += warpSize) 
        py[ic] = __float2half((__half2float(px[ic]) - mean) * __half2float(weight[ic]) * rstd) + bias[ic];
}

class LayerNormPlugin : public IPluginV2DynamicExt{
public:
    float epsilon;
    int axis;

    // construct by creatation
    LayerNormPlugin(float epsilon, int axis){
        this->epsilon = epsilon;
        this->axis    = axis;
    }

    // construct by deserialization
    LayerNormPlugin(const void* data, size_t size){
        const unsigned char* pdata = (const unsigned char*)data;
        this->epsilon = *(float*)pdata;  pdata += sizeof(this->epsilon);
        this->axis    = *((int*)pdata);
    }

    IPluginV2DynamicExt* clone() const noexcept override{
        return new LayerNormPlugin(this->epsilon, this->axis);
    }

    virtual DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override{
        return inputs[0];
    }

    virtual bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override{
        return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) && inOut[pos].type == inOut[0].type;
    }

    virtual void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override{
    } 

    virtual size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override{
        return 0;
    }

    virtual int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept{

        if(inputDesc[0].dims.nbDims != 3){
            printf("Unsupported tensor dims: %d (expected 3)\n", inputDesc[0].dims.nbDims);
            return 1;
        }

        // B, N, C
        int N = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
        int C = inputDesc[0].dims.d[2];
        const void* x      = inputs[0];
        const void* weight = inputs[1];
        const void* bias   = inputs[2];
        void* y            = outputs[0];

        dim3 block(32, 8);
        dim3 grid(1, (N + block.y - 1) / block.y);

        if(inputDesc[0].type == DataType::kHALF){
            layernorm_kernel<half><<<grid, block, 0, stream>>>((half*)x, (half*)weight, (half*)bias, (half*)y, N, C, this->epsilon);
        }else if(inputDesc[0].type == DataType::kFLOAT){
            layernorm_kernel<float><<<grid, block, 0, stream>>>((float*)x, (float*)weight, (float*)bias, (float*)y, N, C, this->epsilon);
        }else{
            // not implemented
            return 1;
        }

        auto code = cudaPeekAtLastError();
        if(code != cudaSuccess){
            printf("Failed to run kernel(layernorm_kernel) with dtype %d\n", (int)inputDesc[0].type);
            return 1;
        }
        return 0;
    }

    virtual nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept{
        return inputTypes[0];
    }

    virtual int32_t initialize() noexcept{
        return 0;
    }

    virtual void terminate() noexcept{

    }

    virtual void serialize(void* buffer) const noexcept{
        unsigned char* pdata = (unsigned char*)buffer;
        *(float*)pdata = this->epsilon;  pdata += sizeof(this->epsilon);
        *(int*)pdata   = this->axis;
    }

    virtual void destroy() noexcept{

    }

    virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept{
    }

    virtual AsciiChar const* getPluginNamespace() const noexcept{
        return "";
    }

    virtual AsciiChar const* getPluginType() const noexcept{
        return "CustomLayerNormalization";
    }

    virtual AsciiChar const* getPluginVersion() const noexcept{
        return "1";
    }

    virtual int32_t getNbOutputs() const noexcept {
        return 1;
    }

    virtual size_t getSerializationSize() const noexcept{
        return sizeof(this->epsilon) + sizeof(this->axis);
    }
};

class LayerNormCreater : public IPluginCreator{
public:
    std::vector<PluginField> fields;
    PluginFieldCollection field_collection;
    std::string namespace_name = "ours";

    LayerNormCreater(){
        fields = {
            PluginField{"epsilon", nullptr, PluginFieldType::kFLOAT32, 1},
            PluginField{"axis",    nullptr, PluginFieldType::kINT32, 1},
        };
        field_collection.fields   = fields.data();
        field_collection.nbFields = fields.size();
    }

    virtual AsciiChar const* getPluginName() const noexcept{
        return "CustomLayerNormalization";
    }

    virtual AsciiChar const* getPluginVersion() const noexcept{
        return "1";
    }

    virtual PluginFieldCollection const* getFieldNames() noexcept{
        return &field_collection;
    }

    virtual IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept{
        assert(strcmp("epsilon", fc->fields[0].name) == 0);
        assert(strcmp("axis",    fc->fields[1].name) == 0);
        float epsilon = *(float*)(fc->fields[0].data);
        int axis      = *(int*)(fc->fields[1].data);
        printf("epsilon = %g, axis=%d\n", epsilon, axis);
        return new LayerNormPlugin(epsilon, axis);
    }

    virtual IPluginV2* deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept{
        return new LayerNormPlugin(serialData, serialLength);
    }

    virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept{
    }

    virtual AsciiChar const* getPluginNamespace() const noexcept{
        return "";
    }
};

REGISTER_TENSORRT_PLUGIN(LayerNormCreater);
