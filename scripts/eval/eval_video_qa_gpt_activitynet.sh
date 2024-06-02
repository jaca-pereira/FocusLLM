EVAL_DATA_DIR=eval/
OUTPUT_DIR=eval/
CKPT_NAME=VideoLLaMA2
CKPT=DAMO-NLP-SG/${CKPT_NAME}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

output_file=${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollama2/eval/run_inference_video_qa_gpt.py \
        --model-path ${CKPT} \
        --video-folder ${EVAL_DATA_DIR}/Activitynet_Zero_Shot_QA/all_test \
        --question-file ${EVAL_DATA_DIR}/Activitynet_Zero_Shot_QA/test_q.json \
        --answer-file ${EVAL_DATA_DIR}/Activitynet_Zero_Shot_QA/test_a.json \
        --output-file ${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode llama_2 &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi


AZURE_API_KEY=your_key
AZURE_API_ENDPOINT=your_endpoint
AZURE_API_DEPLOYNAME=your_deployname

python3 videollama2/eval/eval_video_qa_gpt.py \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/answers/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/answers/${CKPT_NAME}/results.json \
    --api-key $AZURE_API_KEY \
    --api-endpoint $AZURE_API_ENDPOINT \
    --api-deployname $AZURE_API_DEPLOYNAME \
    --num-tasks 4
