#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate focus

echo $CONDA_DEFAULT_ENV

cd ~/data/FocusLLM

set -x

EVAL_DATA_DIR=eval/videochatgpt
OUTPUT_DIR=eval_output/vcgpt
CKPT_NAME=VideoLLaMA2-7B-16F
CKPT=DAMO-NLP-SG/${CKPT_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))



output_file=${OUTPUT_DIR}/answers/correctness/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 -m videollama2.eval.inference_video_oqa_vcgpt_general \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/Test_Videos \
            --question-file ${EVAL_DATA_DIR}/generic_qa.json \
            --answer-file ${OUTPUT_DIR}/answers/correctness/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --focus_layers "16" \
            --focus_segments "1" \
            --reforward True \
            --nr_frames 80 \
            &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/answers/correctness/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done

    mkdir -p ${OUTPUT_DIR}/answers/detail/${CKPT_NAME}
    mkdir -p ${OUTPUT_DIR}/answers/context/${CKPT_NAME}
    cp ${output_file} ${OUTPUT_DIR}/answers/detail/${CKPT_NAME}/merge.json
    cp ${output_file} ${OUTPUT_DIR}/answers/context/${CKPT_NAME}/merge.json
fi

python3 -m videollama2.eval.eval_video_oqa_vcgpt_1_correctness \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/answers/correctness/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/answers/correctness/${CKPT_NAME}/results.json \
    --api-key $OPENAIKEY \
    --num-tasks 16 \


python3 -m videollama2.eval.eval_video_oqa_vcgpt_2_detailed_orientation \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/answers/detail/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/answers/detail/${CKPT_NAME}/results.json \
    --api-key $OPENAIKEY \
    --num-tasks 16 \


python3 -m videollama2.eval.eval_video_oqa_vcgpt_3_context \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/answers/context/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/answers/context/${CKPT_NAME}/results.json \
    --api-key $OPENAIKEY \
    --num-tasks 16 \


output_file=${OUTPUT_DIR}/answers/temporal/${CKPT_NAME}/merge.json

# if output_file not exists then inference
if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 -m videollama2.eval.inference_video_oqa_vcgpt_general \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/Test_Videos \
            --question-file ${EVAL_DATA_DIR}/temporal_qa.json \
            --answer-file ${OUTPUT_DIR}/answers/temporal/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --focus_layers "16" \
            --focus_segments "1" \
            --reforward True \
            --nr_frames 80 \
            &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/answers/temporal/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi


python3 -m videollama2.eval.eval_video_oqa_vcgpt_4_temporal \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/answers/temporal/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/answers/temporal/${CKPT_NAME}/results.json \
    --api-key $OPENAIKEY \
    --num-tasks 16


output_file=${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/merge.json

# if output_file not exists then inference
if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 -m videollama2.eval.inference_video_oqa_vcgpt_consistency \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/Test_Videos \
            --question-file ${EVAL_DATA_DIR}/consistency_qa.json \
            --answer-file ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --focus_layers "16" \
            --focus_segments "1" \
            --reforward True \
            --nr_frames 80 \
            &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi


python3 -m videollama2.eval.eval_video_oqa_vcgpt_5_consistency \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/results.json \
    --api-key $OPENAIKEY \
    --num-tasks 16

