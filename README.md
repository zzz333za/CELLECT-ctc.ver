# CELLECT-ctc.ver-2024.10
# CELLECT (Contrastive Embedding Learning for Large-scale Efficient Cell Tracking)

This project contains the **CELLECT** cell tracking method used for the **Fluo-N3DH-CE** dataset in the Cell Tracking Challenge (CTC).

![CELLECT Image](https://github.com/zzz333za/CELLECT-ctc.ver_2024.10/raw/main/CELLECT.png)

- **CUDA Version**: 12.4  
- **Python Version**: 3.11.7
- **torch**==2.3.1



## Running Instructions
###Install required packages by running:

```bash
pip install -r src/requirements.txt
```

### Data Format and Folder Structure

To perform training, testing, or inference, organize the dataset as follows:

```plaintext
DATA/
└── Fluo-N3DH-CE/
    ├── train/
    │   ├── 01/
    │   ├── 01_GT/
    │   ├── 01_ST/
    │   ├── 02/
    │   ├── 02_GT/
    │   └── 02_ST/
    └── test/
        ├── 01/
        └── 02/
```

Fluo-N3DH-CE: Root folder for the dataset.
train: Folder containing the training data.
01: Raw training images for the first dataset.
01_GT: Ground truth annotations for the first dataset.
01_ST: Silver-standard annotations for the first dataset.
02: Raw training images for the second dataset.
02_GT: Ground truth annotations for the second dataset.
02_ST: Silver-standard annotations for the second dataset.
test: Folder containing the test data.
01: Raw test images for the first test dataset.
02: Raw test images for the second test dataset.

Annotations (GT and ST) Explanation
The GT and ST folders in the dataset contain specific types of annotations used for training and evaluating cell tracking performance:

GT (Ground Truth): This folder contains reference annotations:
- Absolute Truth Corpus: Exact, computer-generated annotations available only for synthetic datasets.
- Gold-Standard Corpus: Human-made annotations, achieved through consensus or majority opinion among several human experts.

ST (Silver Truth): This folder contains computer-generated annotations derived from the majority opinion among results from multiple algorithms submitted by previous challenge participants.

For more details, please refer to the [Cell Tracking Challenge Annotations](https://celltrackingchallenge.net/annotations/).

Place the data within the DATA/Fluo-N3DH-CE folder as specified, and the code will read directly from these directories for training or inference.
### Training the Model

To train the model, use the following command with the specified parameters:
```bash
python ../train.py --data_dir "../../DATA/Fluo-N3DH-CE/train" --out_dir "./Trained models/" --resolution_z 10 --patch_size_xy 256 --patch_size_z 31 --noise 100
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
python ../infer.py --data_dir "../../DATA/Fluo-N3DH-CE/test/02" --out_dir "../../DATA/Fluo-N3DH-CE/02_RES/" --pretrained_weights1 "../../origin_sub/xmodel/U-ext+sc2n-199.0-8.0351.pth" --pretrained_weights2 "../../origin_sub/xmodel/EX+sc2n-199.0-8.0351.pth" --pretrained_weights3 "../../origin_sub/xmodel/EN+sc2n-199.0-8.0351.pth" --resolution_z 10 --patch_size_xy 256 --patch_size_z 31 --overlapxy 128 --overlapz 4
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


**Explanation of pretrained_weights1, pretrained_weights2, and pretrained_weights3**  
The model requires three pretrained weights files, each corresponding to a specific model trained for different tasks in cell tracking:

- **pretrained_weights1**: This is the primary U-Net model used for segmenting regions in the images, identifying cell locations, and extracting cell features. It plays a critical role in initial cell detection.

- **pretrained_weights2**: This is an MLP (Multi-Layer Perceptron) model used to determine if two points within the same frame belong to the same cell. This model aids in identifying and consolidating cell instances within a single frame.

- **pretrained_weights3**: Another MLP model, this one is used to determine if cells across different frames correspond to the same cell, and whether any cell division has occurred. This model enables cell tracking over time and identifies cell divisions when they occur.

**Note:** All three models can be trained from scratch without the need for additional pretrained models. We have included trained versions in the `origin_sub/xmodel/` directory, which you can use directly for comparison or testing purposes in the challenge.

Place the data within the `DATA/Fluo-N3DH-CE` folder as specified, and the code will read directly from these directories for training or inference.
