#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate videollama2

echo $CONDA_DEFAULT_ENV

cd ~/data/VideoLLaMA2

CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval/eval_video_mcqa_videomme.sh