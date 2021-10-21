#!/bin/sh

TASK=$1
MAXLEN=$2
MAX_BATCHSIZE=$3
MIN_BATCHSIZE=$3
GRADACCU=$5
SAVESTEPS=$6
LEARNINGRATE=$7
ALPHA=$8
GUIDE=$9
# python3 run_classification.py  --task_name qnli  --model_type bert  --model_name_or_path bert-base-uncased --data_dir datasets --max_seq_length 128  --per_gpu_trainbatch_size 16  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 16 --learning_rate 3e-5 --save_steps 4096  --num_train_epochs 5  --output_dir qnli_models/bert_base  --do_lower_case  --do_eval  --evaluate_during_training  --do_train
# for imdb and ag:
# --do_test
# DATA_PATH=datasets/${TASK}
DATA_PATH=datasets

echo "Train BERT Model"
mkdir ${TASK}_models
mkdir ${TASK}_models/bert_base
python3 run_classification.py  --task_name ${TASK}  --model_type bert  --model_name_or_path bert-base-uncased --data_dir $DATA_PATH --max_seq_length ${MAXLEN}  --per_gpu_train_batch_size $MAX_BATCHSIZE  --per_gpu_eval_batch_size $MAX_BATCHSIZE --gradient_accumulation_steps $GRADACCU --learning_rate $LEARNINGRATE --save_steps $SAVESTEPS  --num_train_epochs 5  --output_dir ${TASK}_models/bert_base  --do_lower_case  --do_eval  --evaluate_during_training  --do_train

echo "Compute Graident for Residual Strategy"
python3 run_classification.py  --task_name ${TASK}  --model_type bert  --model_name_or_path ${TASK}_models/bert_base --data_dir $DATA_PATH --max_seq_length ${MAXLEN}  --per_gpu_train_batch_size $MIN_BATCHSIZE  --per_gpu_eval_batch_size $MIN_BATCHSIZE  --output_dir ${TASK}_models/bert_base  --do_lower_case  --do_eval_grad

echo "Train the policy network solely"
mkdir ${TASK}_models/auto_1
python3 run_classification.py  --task_name ${TASK}  --model_type autobert  --model_name_or_path ${TASK}_models/bert_base --data_dir $DATA_PATH --max_seq_length ${MAXLEN}  --per_gpu_train_batch_size $MAX_BATCHSIZE  --per_gpu_eval_batch_size $MAX_BATCHSIZE --gradient_accumulation_steps $GRADACCU --learning_rate $LEARNINGRATE --save_steps $SAVESTEPS  --num_train_epochs 3  --output_dir ${TASK}_models/auto_1  --do_lower_case  --do_train --train_rl --alpha $ALPHA --guide_rate $GUIDE

echo "Compute Logits for Knowledge Distilation"
python3 run_classification.py  --task_name ${TASK}  --model_type bert  --model_name_or_path ${TASK}_models/bert_base --data_dir $DATA_PATH --max_seq_length ${MAXLEN}  --per_gpu_train_batch_size $MAX_BATCHSIZE  --per_gpu_eval_batch_size $MAX_BATCHSIZE  --output_dir ${TASK}_models/bert_base  --do_lower_case  --do_eval_logits

echo "Train the whole network with both the task-specifc objective and RL objective"
mkdir ${TASK}_models/auto_1_both
python3 run_classification.py  --task_name ${TASK}  --model_type autobert  --model_name_or_path ${TASK}_models/auto_1 --data_dir $DATA_PATH --max_seq_length ${MAXLEN}  --per_gpu_train_batch_size $MIN_BATCHSIZE  --per_gpu_eval_batch_size $MIN_BATCHSIZE --gradient_accumulation_steps $GRADACCU --learning_rate $LEARNINGRATE --save_steps $SAVESTEPS  --num_train_epochs 3  --output_dir ${TASK}_models/auto_1_both  --do_lower_case  --do_train --train_both --train_teacher --alpha $ALPHA

echo "Evaluate"
python3 run_classification.py  --task_name ${TASK}  --model_type autobert  --model_name_or_path ${TASK}_models/auto_1_both  --data_dir $DATA_PATH --max_seq_length ${MAXLEN}  --per_gpu_train_batch_size $MAX_BATCHSIZE  --per_gpu_eval_batch_size 1  --output_dir ${TASK}_models/auto_1_both  --do_lower_case  --do_eval --eval_all_checkpoints

rm -r npy_folder/ *_models/ logs/