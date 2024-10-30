# CELLECT-ctc.ver-2024.10
# CELLECT (Contrastive Embedding Learning for Large-scale Efficient Cell Tracking)

This project contains the **CELLECT** cell tracking method used for the **Fluo-N3DH-CE** dataset in the Cell Tracking Challenge (CTC).

![CELLECT Image](https://github.com/zzz333za/CELLECT-ctc.ver_2024.10/raw/main/CELLECT.png)

- **CUDA Version**: 12.4  
- **Python Version**: 3.11.7  

## Running Instructions

### Training the Model

To train the model, use the following command with the specified parameters:
```bash
python ../train.py --data_dir "../../../xm1/Fluo-N3DH-CE/train" --out_dir "./Trained models/" --resolution_z 10 --patch_size_xy 256 --patch_size_z 31 --noise 100
```
Training Parameters  
data_dir : Path to the training data folder  
out_dir : Path to save the trained model  
resolution_z : Ratio of z-axis resolution to xy resolution  
patch_size_xy : Size of patches in the xy-plane  
patch_size_z : Size of patches in the z-plane  
noise : Maximum variance of white noise  

### Inference

To run inference, use the following command with the specified parameters:
```bash
python ../infer.py --data_dir "../../../xm1/Fluo-N3DH-CE/test/02" --out_dir "../../../xm1/Fluo-N3DH-CE/02_RES/" --pretrained_weights1 "../../origin_sub/xmodel/U-ext+sc2n-199.0-8.0351.pth" --pretrained_weights2 "../../origin_sub/xmodel/EX+sc2n-199.0-8.0351.pth" --pretrained_weights3 "../../origin_sub/xmodel/EN+sc2n-199.0-8.0351.pth" --resolution_z 10 --patch_size_xy 256 --patch_size_z 31 --overlapxy 128 --overlapz 4
```

Inference Parameters  
data_dir : Path to the test data folder  
out_dir : Path to save the output results  
resolution_z : Ratio of z-axis resolution to xy resolution  
patch_size_xy : Size of patches in the xy-plane  
patch_size_z : Size of patches in the z-plane    
overlapxy : Overlap size in the xy-plane
overlapz : Overlap size in the z-axis  
pretrained_weights : Path to the pretrained model weights  
