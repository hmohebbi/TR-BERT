#!/bin/sh

# TASK, MAXLEN, MAX_BATCHSIZE, MIN_BATCHSIZE, GRADACCU, SAVESTEPS, LEARNINGRATE, ALPHA, GUIDE

./run.sh qnli 128 16 8 16 4090 3e-5 5 0.5 > qnli_A5_G0.5.out
./run.sh qnli 128 16 8 16 4090 3e-5 3 0.5 > qnli_A3_G0.5.out
./run.sh qnli 128 16 8 16 4090 3e-5 2 0.5 > qnli_A2_G0.5.out
./run.sh qnli 128 16 8 16 4090 3e-5 0.1 0.5 > qnli_A0.1_G0.5.out
./run.sh qnli 128 16 8 16 4090 3e-5 0.5 0.5 > qnli_A0.5_G0.5.out