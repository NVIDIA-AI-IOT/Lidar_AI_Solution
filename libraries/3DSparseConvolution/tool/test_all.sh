#!/bin/bash

# SPCONV_CUDA_VERSION=11.4 bash tool/test_all.sh
# SPCONV_CUDA_VERSION=12.6 bash tool/test_all.sh
make pro -j2

cd workspace
./pro fp16 bevfusionXYZ
./pro fp16 bevfusionZYX
./pro fp16 centerpointZYX
python ../tool/compare.py bevfusion/infer.xyz.dense bevfusion/output.xyz.dense --detail
python ../tool/compare.py bevfusion/infer.zyx.dense bevfusion/output.zyx.dense --detail
python ../tool/compare.py centerpoint/out_dense.torch.fp16.tensor centerpoint/output.zyx.dense --detail

./pro int8 bevfusionXYZ
./pro int8 bevfusionZYX
./pro int8 centerpointZYX
python ../tool/compare.py bevfusion/infer.xyz.dense bevfusion/output.xyz.dense --detail
python ../tool/compare.py bevfusion/infer.zyx.dense bevfusion/output.zyx.dense --detail
python ../tool/compare.py centerpoint/out_dense.torch.fp16.tensor centerpoint/output.zyx.dense --detail