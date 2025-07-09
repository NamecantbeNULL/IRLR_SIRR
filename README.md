# Single Image Reflection Removal via Iterative Prompt Learning of Reflection Level 

> **Abstract:** 
Language-guided image restoration has garnered considerable attention and shown promising achievements. However, existing single image reflection removal (SIRR) methods severely overlook text descriptions of reflection levels for the background layer recovery, constraining the reflection removal performance. In this work, we explore learning reflection-level prompts to supervise the optimization of restoration networks in SIRR. Specifically, we propose an Iterative Reflection Level Reduction (IRLR) framework comprising two primary components: the Restoration Network Training Module (RNTM) and the Reflection Level Learning Module (RLLM). Within RNTM, the reflection removal network is trained with learned prompts from RLLM for the estimation of the background layer. While in RLLM, we optimize the learnable prompts to describe the reflection level of an image by harnessing the generated background layer from RNTM. For adaptively supervising the training process of RNTM according to the reflection level of the estimated background layer, a reflection level-aware training strategy is designed for RNTM. To pretrain the learnable prompts for RLLM, we construct a reflection level dataset. Comprehensive experimental results on several released datasets demonstrate that the proposed method significantly outperforms the state-of-the-art methods by 0.74dB PSNR and 0.0122 SSIM.

## Getting started

### Install

We test the code on PyTorch 1.12.1 + CUDA 11.3 + cuDNN 8.3.2.

1. Create a new conda environment
```
conda create -n pt1121 python=3.9
conda activate pt1121
```

2. Install dependencies
```
conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Test

- #### Download test data

Download and unzip the [SIR2+](https://drive.google.com/file/d/17u3zV3aawUVaQFUjt9Fm-fZIk2oFmsKa/view) dataset.
Download and unzip the [Kaggle](https://www.kaggle.com/datasets/siboooo/singleimagereflectionremovaldataset/data) dataset.

- #### Prepare data

The final file path should be the same as the following:

```
── datasets
    └─ SIRR
        └─ test
           ├─ real_I
           │   └─ ... (mixed images)
           └─ real_T
               └─ ... (background images) 

```

- #### Run

You can run 

```python test.py --name IRLR --dataroot ./datasets/SIRR --model IRLR --dataset_mode sirr  --preprocess "" --no_flip --epoch final --gpu_ids 0```	


### Train

- #### Download training data

_Note: Due to the size limit, we are temporarily unable to place the training data we compiled in the supplemental material, which will be open sourced on Github and Google Drive after the double-blind end.

- #### Prepare data

Prepare the training data as:

```
── datasets
    └─ SIRR
        ├─ test
        │  ├─ real_I
        │  │   └─ ... (mixed images)
        │  └─ real_T
        │      └─ ... (background images) 
        └─ train
           ├─ real_I
           │   └─ ... (mixed images)
           └─ real_T
               └─ ... (background images) 
```

- #### Run

You can run 

``` python train.py --dataroot ./datasets/SIRR --name IRLR --model IRCP --dataset_mode sirr --no_flip --gpu_ids 0,1,2,3 --display_id -1 --batch_size 16 --save_epoch_freq 1 --lr 0.0001```	
