# csx4337-project

Student Name: Soumyendu Sarkar 
Student ID: X123160 
Course: COMPSCI X433.7 Machine Learning With TensorFlow 

Project Proposal: Implement DenseNet 121  [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) convolutional neural network model and train both from Scratch and with Imagenet dataset pretrained Transfer Learning to classify the different flower species from images of flowers. 
The Transfer Learned method will reuse the lower convolution layers of the image classifier for their feature extraction capabilities and train a fully connected new classification layer on top to detect different species of flowers. 
In this project the various hyper parameters like learning rate, batch size, and regularization have been tuned to improve model training. 

This project will also include visualization of the training with Tensorboard. 
Data Source : http://download.tensorflow.org/example_images/flower_photos.tgz 
This project has been chosen to demonstrate the power of transfer learning and to show how smaller image datasets can be effectively used by convolutional neural network with limited computation to create a highly accurate image classifier.


# Requirements

Tensorflow 1.8 
Python 3.5+

Install the following Packages :
pip3 install pandas
pip3 install sklearn
pip3 install scipy
pip3 install matplotlib
pip3 install seaborn
pip3 install pyprind
pip3 install pillow

## Pre-trained Models

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN)

Network|Top-1|Top-5|Checkpoints
:---:|:---:|:---:|:---:
DenseNet 121 (k=32)| 74.91| 92.19| [model](https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA)

## Usage

### Step-by-step Example of training on flowers dataset.

#### Downloading ans pre-processing flowers dataset


DATA_DIR=../data/flowers

python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

#### Training of DenseNet 121 model from scratch.
```
DATASET_DIR=../data/flowers
TRAIN_NTL_DIR=./train_scratch_logs
NUM_CLONES=1
python train_image_classifier.py \
    --train_dir=${TRAIN_NTL_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --num_clones=${NUM_CLONES} \
    --model_name=densenet121 
```

#### Training of Pre-Trained DenseNet 121 model with Transfer Learning

```
DATASET_DIR=../data/flowers
TRAIN_TL_DIR=./train_transfer_learning_logs
NUM_CLONES=1
CHECKPOINT_TL_PATH=./transferlearning/tf-densenet121.ckpt

python train_image_classifier.py \
    --train_dir=${TRAIN_TL_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --num_clones=${NUM_CLONES} \
    --model_name=densenet121 \
    --checkpoint_path=${CHECKPOINT_TL_PATH} \
    --checkpoint_exclude_scopes=global_step,densenet121/logits \
    --trainable_scopes=densenet121/logits


#### Validation of Densenet 121 Model Trained from Scratch

DATASET_DIR=../data/flowers
EVAL_DIR=./eval_scratch_logs
CHECKPOINT_M_PATH=./train_scratch_logs
NUM_CLONES=1


python eval_image_classifier.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=densenet121 \
    --num_clones=${NUM_CLONES} \
    --checkpoint_path=${CHECKPOINT_M_PATH}

#### Training of Pre-Trained DenseNet 121 model with Transfer Learning

DATASET_DIR=../data/flowers
EVAL_TL_DIR=./eval_transfer_learning_logs
CHECKPOINT_TLM_PATH=./train_transfer_learning_logs
NUM_CLONES=1

python eval_image_classifier.py \
    --eval_dir=${EVAL_TL_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=densenet121 \
    --num_clones=${NUM_CLONES} \
    --checkpoint_path=${CHECKPOINT_TLM_PATH}

```
### GPU Usage

For use with GPUs set the following flags :

clone_on_cpu to False
num_clones to the number of GPUs

