# **Quantization for CenterPoint SparseConvolution**


<!-- SparseConvolution has two implementations, **[traveller59/spconv](https://github.com/traveller59/spconv)** and **[mmdet3d/spconv](https://github.com/mit-han-lab/bevfusion/tree/main/mmdet3d/ops/spconv)** -->

Centerpoint's SparseConvolution module uses the **[traveller59/spconv](https://github.com/traveller59/spconv)** 

<!-- Centerpoint's SparseConvolution module uses the **[traveller59/spconv](https://github.com/traveller59/spconv)** option

bevfusion's SparseConvolution module uses the **[mmdet3d/spconv](https://github.com/mit-han-lab/bevfusion/tree/main/mmdet3d/ops/spconv)** option -->

### **Install Dependencies**
[PyTorch-Quantization](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) is a toolkit for training and evaluating PyTorch models with simulated quantization.
```bash
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```
### **SparseConvolution QAT Workflow** 
* Insert Q&DQ nodes to get fake-quant pytorch model
* PTQ calibration
* QAT training


### **Notes**

* SparseConvolution has a bias between the facke-quant output and fp32 output. This bias may amplify the error after Bn processing.So fuse-bn during QAT training can effectively reduce this effect.

* When exporting onnx, fuse-relu is needed to get a better performance.

* The quant of add is to count the dynamic range of the input to the add node. Used in network inference.

* Sensitivity profile can analyse the impact of each layer on accuracy, If a layer has a large mAP drop, you can disable the quantization of this layer.

<br>

### **Usage**
Clone And Apply Patch

- Git clone [CenterPoint](https://github.com/tianweiy/CenterPoint) and install Dependencies, Please refer [INSTALL](https://github.com/tianweiy/CenterPoint/blob/master/docs/INSTALL.md) to set up libraries 
```bash
git clone https://github.com/tianweiy/CenterPoint.git
```

- Apply this patch to your centerpoint project
```bash
cp -r  * CenterPoint/
```
### **Sensitivity Profile**

If you want to know which quant layers have a big impact on the results, please run 
```bash
python tools/centerpoint_eval.py  sensitivity --nuScense-config=./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py --weight=workspace/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/epoch_20.pth  --calibrate_batch 400
```
### **Workflow in PTQ Mode**
If you want to quickly verify the effect of model quantization, you can try the PTQ mode.

1.Generate PTQ model
```bash
python -m torch.distributed.launch --nproc_per_node=1  ./tools/centerpoint_qat_pipline.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py --resume_from=workspace/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/epoch_20.pth --calibrate_batch=400 --work_dir=workspace/PTQ_QAT --ptq_mode
```

2.Evaluation PTQ mAP

```bash
python tools/centerpoint_eval.py  test --nuScense-config=./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py --weight=./workspace/snapshot/2023_02_18_00_00_53/model_202302171.pth --use-quantization  
```

3.Export Model to Onnx
```bash
python onnx_export/export-scn.py --ckpt=ptq.pth --save-onnx=ptq.onnx --use-quantization
```

### **Workflow in QAT Mode** 
If you want to achieve better quantization results, you can try the QAT mode. But this will take longer time for model finetune.

1.Generate QAT model<br />
- Modify the learning rate in config file
(configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py) 
```bash
lr_config = dict(
    type="one_cycle", lr_max=0.0001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)
```
- Run QAT training for centerpoint
```bash
python -m torch.distributed.launch --nproc_per_node=1  ./tools/centerpoint_qat_pipline.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py --resume_from=workspace/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/epoch_20.pth --calibrate_batch=400 --work_dir=workspace/PTQ_QAT
```

2.Evaluation & Export command is the same as PTQ

### **Verification**
Centerpoint SparseConvolution PTQ/QAT Performance on nuScenes Validation dataset
| Model  |  Validation MAP |  Validation NDS |  Calibration method | Pytorch weight | Onnx weight|
|---|---|---|---|---|---|
| pytorch fp16  | 59.55   |  66.75  |  /  |  [epoch20.pth](https://drive.google.com/file/d/1ujuhCXA7QFLrRALvm-TqxbPOL8pfBzAO/view?usp=share_link)  |  [epoch20.onnx](https://drive.google.com/file/d/1zdWdxBuIeOdgkc7uTNRv0khxiPhGi8Bg/view?usp=share_link)  |
| pytorch PTQ FakeQuant  | 59.08  |  66.45 | Histogram   |   [ptq.pth](https://drive.google.com/file/d/1DciBfi69O6EDmMbPi13u41fSCN5EZ_sp/view?usp=sharing)  |  [ptq.onnx](https://drive.google.com/file/d/1w8KzzHvLtQ_d0FhmaVzpdxckdROjcvEB/view?usp=sharing)  |   
| pytorch QAT FakeQuant  | 59.20  |  66.53 | Histogram |  [qat.pth](https://drive.google.com/file/d/1qcKhDq4WVtmLRGZrSV1rM3e8adRuEqM2/view?usp=share_link)  |  [qat.onnx](https://drive.google.com/file/d/1veSFFVZx1PGLE2ODH8v5Yf-H7uNooJJs/view?usp=share_link)  |
