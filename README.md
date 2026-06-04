# Single Image Reflection Removal via Iterative Prompt Learning of Reflection Level 

> **Abstract:** 
Single-image reflection removal (SIRR) aims to restore the latent background layer from a reflection-contaminated image. Despite the promising progress achieved by deep learning-based methods, the role of negative training samples and descriptive prompts for the reflection severity is underexplored in most existing deep SIRR approaches, limiting their reflection removal performance and generalization capability. In this work, we introduce a novel training framework that synergistically leverages learnable prompts and image data to optimize the restoration network. To this end, we define reflection levels corresponding to varying degrees of reflection interference on the background content and learn reflection-level prompts to supervise the SIRR process. We propose an Iterative Reflection Level Reduction (IRLR) framework composed of a Restoration Network Training Module (RNTM) and a Reflection Level Learning Module (RLLM). Specifically, RNTM predicts the background layer under the guidance of prompts learned by RLLM, while RLLM in turn refines these prompts using outputs from RNTM. The two modules are trained iteratively to progressively reduce the reflection levels of estimated background layers. To initialize the prompts, we construct a dedicated reflection-level dataset for pretraining. For adaptively supervising RNTM, we design a new reflection-level-aware strategy to address the challenge of directly aligning the output background with the minimal reflection level. Comprehensive experimental results on several released datasets demonstrate that the proposed method significantly outperforms the state-of-the-art methods by 0.74dB PSNR and 0.0122 SSIM.

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


## Citations
#### BibTeX

    @article{song2026irlr,
      author={Song, Binbin and Zhou, Jiantao and Xu, Shuning and Liu, Xina and Wu, Haiwei and Fan, Xiaopeng and Wen, Bihan},
      journal={IEEE Transactions on Image Processing}, 
      title={Single-Image Reflection Removal via Iterative Prompt Learning of Reflection Level}, 
      year={2026},
      volume={35},
      number={},
      pages={5698-5713}}
