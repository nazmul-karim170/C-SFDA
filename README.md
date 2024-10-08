
<h2 align="center"> <a href="https://github.com/nazmul-karim170/C-SFDA_Source-Free-Domain-Adaptation/tree/main">C-SFDA: A Curriculum Learning Aided Self-Training Framework for Efficient
Source Free Domain Adaptation</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2312.09313-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2303.17132)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/nazmul-karim170/C-SFDA_Source-Free-Domain-Adaptation/blob/main/LICENSE) 


</h5>

## [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karim_C-SFDA_A_Curriculum_Learning_Aided_Self-Training_Framework_for_Efficient_Source_CVPR_2023_paper.pdf) 


## First, Create an Environment
	
	conda create -n domain_ada 
	conda activate domain_ada
	
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
	pip install hydra-core numpy omegaconf sklearn tqdm wandb

		
## For Training on VisDA-C

### **Prepare dataset**

Please download the [VisDA-C dataset](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), and put it under `${DATA_ROOT}`. For your convenience we also compiled `.txt` files based on the the image labels, provided under `./data/VISDA-C/`. The prepared directory would look like:

```bash
${DATA_ROOT}
├── VISDA-C
│   ├── train
│   ├── validation
│   ├── train_list.txt
│   ├── validation_list.txt
```

`${DATA_ROOT}` is set to `./data/` by default, which can be modified in `configs/data/basic.yaml` or via hydra command line interface `data.data_root=${DATA_ROOT}`.

### **Training**
We use [hydra](https://github.com/facebookresearch/hydra) as the configuration system. By default, the working directory is `./output`, which can be changed directly from `configs/root.yaml` or via hydra command line interface `workdir=${WORK_DIR}`.

VISDA-C experiments are done for `train` to `validation` adaptation. Before the test-time adaptation, we should have the source model. You may train the source model with script `train_source_VisDA.sh` as shown below.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash train_source_VisDA.sh 
```

We also provide the pre-trained source models from 3 seeds (2020, 2021, 2022) which can be [downloaded from here](https://drive.google.com/drive/folders/16vTNNzzAt4M1mmeLsOxSFDRzBogaNkJw?usp=sharing).

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash train_target_VisDA.sh 
```

### Note
If the simulation ends without any error, set HYDRA_FULL_ERROR=1
GPU Usage: 2 NVIDIA RTX A40 GPUs

## Reference

If you find this work helpful to your own work, please consider citing us:
```
@inproceedings{karim2023c,
  title={C-sfda: A curriculum learning aided self-training framework for efficient source free domain adaptation},
  author={Karim, Nazmul and Mithun, Niluthpol Chowdhury and Rajvanshi, Abhinav and Chiu, Han-pang and Samarasekera, Supun and Rahnavard, Nazanin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24120--24131},
  year={2023}
}
```

