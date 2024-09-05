import os
import re
import math
import json
import argparse
import warnings

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

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

class VCGPTDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, video_folder, data_list, processor):
        self.video_folder = video_folder
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        question = line['Q']
        answer = line['A']
        video_name = line['video_name']

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        video_tensor = self.processor(video_path)

        return {
            'video': video_tensor,
            'video_name': video_name,
            'question': question,
            'answer': answer,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    v_id = [x['video_name'] for x in batch]
    qus = [x['question'] for x in batch]
    ans = [x['answer'] for x in batch]
    vid = torch.stack(vid, dim=0)
    return vid, v_id, qus, ans


def run_inference(args):
    # Initialize the model
    model, processor, tokenizer, version = model_init(args.model_path, args.focus_layers, args.focus_segments, args.reforward, args.nr_frames)

    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = VCGPTDataset(args.video_folder, questions, processor)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Iterate over each sample in the ground truth file
    for i, (video_tensors, video_names, questions, answers) in enumerate(tqdm(dataloader)):

        # reduce batch dimension
        video_tensor = video_tensors[0]
        video_name = video_names[0]
        question = questions[0]
        answer = answers[0]

        output = x_infer(
            video_tensor,
            question, 
            mode='vanilla',
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            version=version,
        )

        qa = {'video_name': video_name, 'Q': question, 'A': answer, 'P': output}

        ans_file.write(json.dumps(qa) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument('--focus_layers', help='Focus layers for the model.', required=True, type=str)
    parser.add_argument('--focus_segments', help='Focus segments for the model.', required=True, type=str)
    parser.add_argument('--reforward', help='Reforward parameter for the model.', required=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--nr_frames', help='Number of frames to process.', required=True, type=int)

    args = parser.parse_args()

    run_inference(args)
