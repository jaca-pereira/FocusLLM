#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate focus

echo $CONDA_DEFAULT_ENV

cd ~/data/FocusLLM

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

focus_layers_list=(
3, 16, 28, 3, 16, 28,
3, 16, 28, 3, 16, 28,
3, 16, 28, 3, 16, 28, (3, 16), (16,28),
3, 16, 28, 3, 16, 28, (3, 16), (16,28),
3, 16, 28, 3, 16, 28, (3, 16), (16,28), (3, 16, 28),
3, 16, 28, 3, 16, 28, (3, 16), (16,28), (3, 16, 28), (3, 16, 28))

focus_segments_list=(
1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, (2,1), (2,1)
1, 1, 1, 1, 1, 1, (2,1), (2,1),
1, 1, 1, 1, 1, 1, (3,1), (3,1), (4, 2, 1),
1, 1, 1, 1, 1, 1, (3,1), (3,1), (4, 2, 1), (5, 3, 1))

reforward_list=(
true, true, true, false, false, false,
true, true, true, false, false, false,
true, true, true, false, false, false, false, false
true, true, true, false, false, false, false, false,
true, true, true, false, false, false, false, false, false,
true, true, true, false, false, false, false, false, false, false)

nr_frames_list=(
32, 32, 32, 32, 32, 32,
48, 48, 48, 48, 48, 48,
64, 64, 64, 64, 64, 64, 64, 64,
80, 80, 80, 80, 80, 80, 80, 80,
96, 96, 96, 96, 96, 96, 96, 96, 96,
128, 128, 128, 128, 128, 128, 128, 128, 128, 128)


for i in {0..49}
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/eval_video_mcqa_videomme.sh --focus_layers "${focus_layers_list[i]}" --focus_segments "${focus_segments_list[i]}" --reforward "${reforward_list[i]}" --nr_frames "${nr_frames_list[i]}"
done
