#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate focus

echo $CONDA_DEFAULT_ENV

cd ~/data/FocusLLM

CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval/eval_video_mcqa_videomme.sh
