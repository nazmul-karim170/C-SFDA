
## This Repo is still under Construction!! There may be issues!


# First, Create an Environment
	
	conda create -n domain_ada 
	conda activate domain_ada
	
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
	pip install hydra-core numpy omegaconf sklearn tqdm wandb

		
# Download the dataset: VISDA-C

1. For VISDA-C, go to this link [Link](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and download the train.tar, validation.tar and test.tar

Put these downloaded files into "data/VISDA-C/" folder.
The text files should be in the proper folder, expercially validation_list.txt file. 
	
	
# Download the source model 
1. For Downloading the source models trained on VISDA-C from here [Link](https://drive.google.com/drive/folders/1Uf4jCsGX0WcC8aHstdEG7FvCR6DBSufk?usp=sharing)

2. Put them in the "checkpoint" folder. 

# Run Commands
	
1. For VISDA-C dataset, for adapting a model from "train" to "val"
 
 		export CUDA_VISIBLE_DEVICES=0,1,2,3
		bash train_target_VisDA.sh 
		
# To check on our proposed algorithm, please go to "target_visda.py" and check "train_csfda" function.

Note: 1. If the simulation ends without any error, set HYDRA_FULL_ERROR=1
GPU Usage: 2 NVIDIA RTX A40 GPUs

