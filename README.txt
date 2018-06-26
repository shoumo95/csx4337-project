Requires :
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



Step-by-step Example of training on flowers dataset.

Downloading ans converting flowers dataset

DATA_DIR=../data/flowers

python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"



Training of Pre-Trained DenseNet 121 model with Transfer Learning

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




Training of DenseNet 121 model from scratch.

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



Validation of Densenet 121 Model Trained from Scratch

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



Validation of Transfer Learned Densenet 121 Model

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



GPU Usage

For use with GPUs set the following flags :

clone_on_cpu to False
num_clones to the number of GPUs

