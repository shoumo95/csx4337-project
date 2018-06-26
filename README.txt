Requires :
  Tensorflow 1.8 
  Python 3.5+


Step-by-step Example of training on flowers dataset.

Downloading ans converting flowers dataset

DATA_DIR=../data/flowers

python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"



Transfer Learning from Pre-Trained DenseNet 121 model parameters 

DATASET_DIR=../data/flowers
TRAIN_TL_DIR=./train_transfer_learning_logs
NUM_CLONES=2
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




Training a model from scratch.

DATASET_DIR=../data/flowers
TRAIN_NTL_DIR=./train_scratch_logs
NUM_CLONES=2
python train_image_classifier.py \
    --train_dir=${TRAIN_NTL_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --num_clones=${NUM_CLONES} \
    --model_name=densenet121 





export DATA_HOME=${PWD}/data
export CODE_HOME=${PWD}
export DATA_DIR=${DATA_HOME}/flowers
export TRAIN_DIR=${CODE_HOME}/train_logs
export NUM_CLONES=2
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATA_DIR} \
    --model_name=densenet121 \
    --num_clones=${NUM_CLONES}


Validation

DATASET_DIR=../data/flowers
EVAL_DIR=./eval_scratch_logs
CHECKPOINT_M_PATH=./train_scratch_logs
NUM_CLONES=2


python eval_image_classifier.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=densenet121 \
    --num_clones=${NUM_CLONES} \
    --checkpoint_path=${CHECKPOINT_M_PATH}



#!/usr/bin/env bash
export DATA_HOME=${PWD}/data
export CODE_HOME=${PWD}
export DATA_DIR=${DATA_HOME}/flowers
export TRAIN_DIR=${CODE_HOME}/train_logs
export NUM_CLONES=2

# to evaluate a specific checkpoint
export CHECKPOINT_PATH_EVAL=${TRAIN_DIR}

export EVAL_DIR=${CODE_HOME}/eval_logs

python eval_image_classifier.py \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --dataset_dir=${DATA_DIR} \
    --model_name=densenet121 \
    --checkpoint_path=${CHECKPOINT_PATH_EVAL} \
    --eval_dir=${EVAL_DIR}







Validation

DATASET_DIR=../data/flowers
EVAL_DIR=./eval_logs
CHECKPOINT_M_PATH=./train_transfer_learning_logs/model.ckpt-1888


python eval_image_classifier.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=densenet121 \
    --checkpoint_path=${CHECKPOINT_TM_PATH}


