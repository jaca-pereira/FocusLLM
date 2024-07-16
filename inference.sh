#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate videollama2

echo $CONDA_DEFAULT_ENV

cd ~/data/VideoLLaMA2

python3 inference.py