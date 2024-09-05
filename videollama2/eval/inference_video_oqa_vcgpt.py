import os
import json
import math
import argparse
import warnings

import numpy as np
from tqdm import tqdm

import sys
sys.path.append('./')
from videollama2 import model_init, x_infer

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq

def run_inference(args):
    # Initialize the model
    model, processor, tokenizer, version = model_init(args.model_path, args.focus_layers, args.focus_segments, args.reforward, args.nr_frames)

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.answer_file, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for idx, sample in enumerate(tqdm(gt_questions)):
        video_name = sample['video_name']
        question = sample['question']
        qid = sample['question_id']
        answer = gt_answers[idx]['answer']

        # Load the video file
        for fmt in video_formats:
            temp_path = os.path.join(args.video_folder, f"v_{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
            # BUG: compatibility for MSVD, MSRVTT, TGIF
            temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # question = question + '\n' + 'Answer the question using a single word or a short phrase with multiple words.'

        video_tensor = processor(video_path)
        output = x_infer(
            video_tensor,
            question, 
            mode='vanilla',
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            version=version,
        )

        sample_set = {'id': qid, 'question': question, 'answer': answer, 'pred': output}
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--focus_layers', help='Focus layers for the model.', required=True, type=str)
    parser.add_argument('--focus_segments', help='Focus segments for the model.', required=True, type=str)
    parser.add_argument('--reforward', help='Reforward parameter for the model.', required=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--nr_frames', help='Number of frames to process.', required=True, type=int)

    args = parser.parse_args()

    run_inference(args)
