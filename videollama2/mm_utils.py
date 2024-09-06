import ast
import itertools
import math
import base64
import os
import pickle
from io import BytesIO

import torch
import decord
import imageio
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from matplotlib import pyplot as plt
import seaborn as sns
from moviepy.editor import VideoFileClip
from scipy.stats import ks_2samp, entropy
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from transformers import StoppingCriteria
import torch.nn.functional as F
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.stats_manager import StatsManager
from tqdm import tqdm as tqdm
from .constants import NUM_FRAMES, MAX_FRAMES, NUM_FRAMES_PER_SECOND, MMODAL_INDEX_TOKEN, IMAGE_TOKEN_INDEX


def merge_scenes(cut_list, cut_scores, scene_list,num_frames,max_scene_num=4, num_frame_per_scene=8, min_frames_per_scene=30):
    if len(scene_list) == len(cut_list) and len(scene_list) == 0:
        frame_ids = np.linspace(0, num_frames-1, num_frame_per_scene, dtype=int)  # only one scene for current video
        return [frame_ids]

    scene_list, cut_results = merge_scenes_not_exeed_max_scene_num(cut_list,cut_scores,scene_list, max_scene_num)

    prev_cut_point = 0
    list_of_scene_frames = [] 
    for (cur_cut_point, _) in cut_results:
        frame_ids = list(np.linspace(prev_cut_point, cur_cut_point-1, num_frame_per_scene, dtype=int))
        list_of_scene_frames.append(frame_ids)
        prev_cut_point = cur_cut_point
    if cur_cut_point < num_frames:
        frame_ids = np.linspace(cur_cut_point, num_frames-1, num_frame_per_scene, dtype=int)
        list_of_scene_frames.append(frame_ids)

    return list_of_scene_frames


def merge_scenes_not_exeed_max_scene_num(cut_list,cut_scores, scene_list, max_scene_num):
    cut_frames = [ele.get_frames() for ele in cut_list]
    cut_results = list(zip(cut_frames, cut_scores))
    while len(scene_list) > max_scene_num:
        min_idx = np.argmin(cut_scores)
        cut_frames = [ele for idx, ele in enumerate(cut_frames) if idx != min_idx]
        cut_scores = [ele for idx, ele in enumerate(cut_scores) if idx != min_idx]

        # merge scene list
        num_scenes = len(scene_list)
        #print("Current min_idx:", min_idx)
        s1 = scene_list[min_idx]
        s2 = scene_list[min_idx+1]
        new_scene = (s1[0], s2[1])
        if min_idx == 0:
            # merge the first two scenes
            new_scene_list = [new_scene] + scene_list[2:]
        elif min_idx == num_scenes - 1:
            # # merge the last two scenes
            new_scene_list = scene_list[:min_idx-1] + [new_scene]
        else:
            new_scene_list = scene_list[:min_idx] + [new_scene] + scene_list[min_idx+2:]
        scene_list = new_scene_list
        cut_results = list(zip(cut_frames, cut_scores))
    return scene_list, cut_results


def split_video_into_scenes(video_path, threshold=27.0, max_scene_num=10, num_frame_per_scene=8):
    # Open video, create a scene manager, and add a detector.
    video = open_video(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    detector = ContentDetector(threshold=threshold)
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    cut_list = scene_manager.get_cut_list()
    num_frames = video.duration.get_frames()
    if len(scene_list) == len(cut_list) and len(scene_list) == 0:
        frame_ids = np.linspace(0, num_frames-1, num_frame_per_scene, dtype=int)  # only one scene for current video
        return [frame_ids]
    assert len(scene_list) == len(cut_list) + 1, f"inconsistent lengths for scene list ({len(scene_list)}) vs. cut list ({len(cut_list)})"
    cut_frames = [ele.get_frames() for ele in cut_list]
    cut_scores = [stats_manager.get_metrics(f, ["delta_lum"])[0] for f in cut_frames]
    cut_results = list(zip(cut_frames, cut_scores))
    #print(f"Original cut scores: {cut_scores}, original scene list: {scene_list}")
    while len(scene_list) > max_scene_num:
        min_idx = np.argmin(cut_scores)
        cut_frames = [ele for idx, ele in enumerate(cut_frames) if idx != min_idx]
        cut_scores = [ele for idx, ele in enumerate(cut_scores) if idx != min_idx]

        # merge scene list
        num_scenes = len(scene_list)
        #print("Current min_idx:", min_idx)
        s1 = scene_list[min_idx]
        s2 = scene_list[min_idx+1]
        new_scene = (s1[0], s2[1])
        if min_idx == 0:
            # merge the first two scenes
            new_scene_list = [new_scene] + scene_list[2:]
        elif min_idx == num_scenes - 1:
            # # merge the last two scenes
            new_scene_list = scene_list[:min_idx-1] + [new_scene]
        else:
            new_scene_list = scene_list[:min_idx] + [new_scene] + scene_list[min_idx+2:]
        scene_list = new_scene_list
        cut_results = list(zip(cut_frames, cut_scores))
    #print(f"Cut scores after merging: {cut_scores}, scene list: {scene_list}")
    prev_cut_point = 0
    list_of_scene_frames = [] 
    for (cur_cut_point, _) in cut_results:
        frame_ids = list(np.linspace(prev_cut_point, cur_cut_point-1, num_frame_per_scene, dtype=int))
        list_of_scene_frames.append(frame_ids)
        prev_cut_point = cur_cut_point
    if cur_cut_point < num_frames:
        frame_ids = np.linspace(cur_cut_point, num_frames-1, num_frame_per_scene, dtype=int)
        list_of_scene_frames.append(frame_ids)
    # print(f"Finally got {len(list_of_scene_frames)} scenes where we evenly sampled {num_frame_per_scene} frames for each scene")
    return list_of_scene_frames


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.
    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')
    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution
        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)
    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.
    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.
    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)
    # Resize the image
    resized_image = image.resize((new_width, new_height))
    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.
    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.
    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches


def get_anyres_image_grid_shape(image_size, grids, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.
    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grids (str, List[tuple[int]]): Patch segmentation grid.
        patch_size (int): The size of each image patch.
    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grids) is list:
        possible_resolutions = [(x * patch_size, y * patch_size) for x, y in grids]
    else:
        possible_resolutions = [(x * patch_size, y * patch_size) for x, y in ast.literal_eval(grids)]
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, grids, patch_size):
    """
    Process an image with variable resolutions.
    Args:
        image (PIL.Image.Image): The input image to be processed.
        grids (str, List[tuple[int]]): Patch segmentation grid.
        patch_size (int): The size of the patches to be extracted.
    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grids) is list:
        possible_resolutions = [(x * patch_size, y * patch_size) for x, y in grids]
    else:
        possible_resolutions = [(x * patch_size, y * patch_size) for x, y in ast.literal_eval(grids)]
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)
    patches = divide_to_patches(image_padded, patch_size)
    image_original_resize = resize_and_pad_image(image, (patch_size, patch_size))
    image_patches = [image_original_resize] + patches
    return image_patches


def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def frame_expansion(frame_list, n):
    assert len(frame_list) == n * n
    width, height = frame_list[0].width, frame_list[0].height
    expanded_width = n * width
    expanded_height = n * height
    expanded_frame = Image.new('RGB', (expanded_width, expanded_height))
    for i in range(n):
        for j in range(n):
            frame = frame_list[i * n + j]
            coordinate = (j*width, i*height)
            expanded_frame.paste(frame, coordinate)
    return expanded_frame


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    #print("Current image_aspect_ratio:", image_aspect_ratio)
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def process_videos(frames, image_processor, model_cfg):
    # this function only used during inference
    # image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    # new_frames = []
    # print("Current image_aspect_ratio:", image_aspect_ratio)
    # if image_aspect_ratio == 'pad':
    #     for image in frames:
    #         image = Image.fromarray(image)
    #         image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    #         image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    #         new_frames.append(image)
    # else:
    #     return image_processor(frames, return_tensors='pt')['pixel_values']
    # if all(x.shape == new_frames[0].shape for x in new_frames):
    #     new_frames = torch.stack(new_frames, dim=0)
    new_frames = image_processor.preprocess(frames, return_tensors='pt')['pixel_values']  # do not pad for video frames
    return new_frames


def create_photo_grid(arr, rows=None, cols=None):
    """
    Create a photo grid from a 4D numpy array with shape [t, h, w, c].

    Parameters:
        arr (numpy.ndarray): Input array with shape [t, h, w, c].
        rows (int): Optional. Number of rows in the grid. If not set, it will be determined based on `cols` or the square root of `t`.
        cols (int): Optional. Number of columns in the grid. If not set, it will be determined based on `rows` or the square root of `t`.

    Returns:
        numpy.ndarray: A 3D numpy array representing the photo grid.
    """

    if isinstance(arr, list):
        if isinstance(arr[0], Image.Image):
            arr = np.stack([np.array(img) for img in arr])
        elif isinstance(arr[0], np.ndarray):
            arr = np.stack(arr)
        else:
            raise ValueError("Invalid input type. Expected list of Images or numpy arrays.")

    t, h, w, c = arr.shape
    
    # Calculate the number of rows and columns if not provided
    if rows is None and cols is None:
        rows = math.ceil(math.sqrt(t))
        cols = math.ceil(t / rows)
    elif rows is None:
        rows = math.ceil(t / cols)
    elif cols is None:
        cols = math.ceil(t / rows)

    # Check if the grid can hold all the images
    if rows * cols < t:
        raise ValueError(f"Not enough grid cells ({rows}x{cols}) to hold all images ({t}).")
    
    # Create the grid array with appropriate height and width
    grid_height = h * rows
    grid_width = w * cols
    grid = np.zeros((grid_height, grid_width, c), dtype=arr.dtype)
    
    # Fill the grid with images
    for i in range(t):
        row_idx = i // cols
        col_idx = i % cols
        grid[row_idx*h:(row_idx+1)*h, col_idx*w:(col_idx+1)*w, :] = arr[i]
    
    return grid


def process_image(image_path, processor, aspect_ratio='pad', num_frames=NUM_FRAMES, image_grid=False):
    image = Image.open(image_path).convert('RGB')

    if image_grid:
        pg = np.stack([np.array(image)] * num_frames)
        grid_h = grid_w = math.ceil(math.sqrt(num_frames))
        pg = create_photo_grid(pg, grid_h, grid_w)
        images = [pg, np.array(image)]
    else:
        images = [np.array(image)]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return images


def process_video(video_path, processor, aspect_ratio='pad', num_frames=NUM_FRAMES, image_grid=False, sample_scheme='uniform'):
    def frame_sample(duration, mode='uniform', local_fps=None):
        if mode == 'uniform':
            # Calculate the size of each segment from which a frame will be extracted
            seg_size = float(duration - 1) / num_frames

            frame_ids = []
            for i in range(num_frames):
                # Calculate the start and end indices of each segment
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                # Append the middle index of the segment to the list
                frame_ids.append((start + end) // 2)

            return frame_ids
            # NOTE: old version
            # return np.linspace(0, duration-1, num_frames, dtype=int)
        elif mode == 'fps':
            assert local_fps is not None
            segment_len = min(local_fps // NUM_FRAMES_PER_SECOND, duration)
            return np.arange(segment_len // 2, duration, segment_len, dtype=int)
        else:
            raise ImportError(f'Unsupported frame sampling mode: {mode}')

    if isinstance(video_path, str):
        if video_path.endswith('.gif'):
            video_gif = imageio.get_reader(video_path)
            duration, local_fps = len(video_gif), 10

            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)
            video_data = [frame for index, frame in enumerate(video_gif) if index in frame_id_list]
        # added by lixin4ever, include the support of .webm files from sthsthv2
        elif video_path.endswith('.webm'):
            video_webm = VideoFileClip(video_path)
            video_frames = np.array(list(video_webm.iter_frames()))

            duration, local_fps = len(video_frames), video_webm.fps

            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)
            video_data = video_frames[frame_id_list]
        else:
            # NOTE: num_threads=1 is required to avoid deadlock in multiprocessing
            decord_vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) 
            duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
        
            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)

            try:
                video_data = decord_vr.get_batch(frame_id_list).numpy()
            except:
                video_data = decord_vr.get_batch(frame_id_list).asnumpy()


            # if self.data_args.use_temp_aug:
            #     frame_id_list = np.linspace(0, duration-1, num_frames * 2 * 2, dtype=int)
            #     video_data = decord_vr.get_batch(frame_id_list)
            #     video_frames = [Image.fromarray(f) for f in video_data.numpy()]
            #     chunked_video_frames = chunk_list(video_frames, 2*2)
            #     video_data = [frame_expansion(frame_list, 2) for frame_list in chunked_video_frames]
    elif isinstance(video_path, np.ndarray):
        assert len(video_path) == num_frames
        video_data = video_path
    elif isinstance(video_path, list):
        assert len(video_path) == num_frames
        video_data = np.stack([np.array(x) for x in video_path])

    if image_grid:
        grid_h = grid_w = math.ceil(math.sqrt(num_frames))
        pg = create_photo_grid(video_data, grid_h, grid_w)
        video_data = [pg, *video_data]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    else:
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']

    return video


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(f'<{MMODAL_INDEX_TOKEN[MMODAL_token_index].lower()}>')]
    num_prompt_chunks = len(prompt.split(f'<{MMODAL_INDEX_TOKEN[MMODAL_token_index].lower()}>'))

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [MMODAL_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)

def visualize_hidden_states(all_hidden_states, modal_token_position, num_image_video_tokens, filename, video_path, prompt):
    start_image_video = modal_token_position + 1
    end_image_video = start_image_video + num_image_video_tokens
    hidden_states = torch.cat(all_hidden_states[0], 0)
    norm_hidden_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True, p=2)
    similarity_matrix = torch.matmul(norm_hidden_states, norm_hidden_states.transpose(1, 2)).cpu().detach().numpy()
    max_sim = np.max(similarity_matrix)
    min_sim = np.min(similarity_matrix)
    print(f'Maximum similarity: {max_sim}')
    print(f'Minimum similarity: {min_sim}')
    os.makedirs('./figures', exist_ok=True)
    os.makedirs(f'./figures/{filename}', exist_ok=True)
    os.makedirs(f'./figures/{filename}/hidden_states', exist_ok=True)
    os.makedirs(f'./figures/{filename}/hidden_states/{video_path}', exist_ok=True)
    os.makedirs(f'./figures/{filename}/hidden_states/{video_path}/{prompt}', exist_ok=True)
    for layer_nr, sim in enumerate(similarity_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim, annot=False, cmap='coolwarm', vmin=min_sim, vmax=max_sim)
        plt.axvline(start_image_video + 1, color='green', linestyle='dashed', linewidth=1, label='begin_image_video')
        plt.axvline(end_image_video, color='green', linestyle='dashed', linewidth=1,
                    label='end_image_video')
        plt.axhline(start_image_video, color='green', linestyle='dashed', linewidth=1)
        plt.axhline(end_image_video, color='green', linestyle='dashed', linewidth=1)
        # legend
        plt.legend(loc='upper right')
        plt.title(f'Cosine Similarity Matrix for layer {layer_nr}')
        plt.xlabel('Token Position')
        plt.ylabel('Token Position')
        plt.savefig(f'./figures/{filename}/hidden_states/{video_path}/{prompt}/layer_{layer_nr}.png')
        plt.close()


def visualize_hidden_states_distribution(all_hidden_states, filename, video_path, prompt):
    hidden_states = torch.cat(all_hidden_states[0], 0)
    norm_hidden_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True, p=2)
    similarity_matrix = torch.matmul(norm_hidden_states, norm_hidden_states.transpose(1, 2)).cpu().detach().numpy()
    #save the distribution of cosine similarity for each layer and the final.
    pkl_path = f'./figures/{filename}/hidden_states/{video_path}/{prompt}/similarity_matrix.pkl'
    pickle.dump(similarity_matrix, open(pkl_path, 'wb'))
    # Determine grid size
    num_layers = len(norm_hidden_states)
    grid_size = math.ceil(math.sqrt(num_layers))  # Square root to find grid size
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
    fig.suptitle('Cosine Similarity Distribution Across Layers')

    for i, cos_sim in enumerate(tqdm(similarity_matrix)):
        row = i // grid_size
        col = i % grid_size
        if grid_size == 1:  # If there's only one subplot, axs is not a 2D array
            ax = axs
        else:
            ax = axs[row, col]
        ax.hist(cos_sim.flatten(), bins=10, alpha=0.75)
        ax.set_title(f'Layer {i + 1}')
        ax.set_xlim([-1, 1])

    # Hide empty subplots
    for i in range(num_layers, grid_size ** 2):
        row = i // grid_size
        col = i % grid_size
        if grid_size > 1:
            axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f'./figures/{filename}/hidden_states/{video_path}/{prompt}/distribution.png')
    plt.close()

def visualize_average_attention(attentions, modal_token_position, num_image_video_tokens, filename, video_path, prompt, min_attention=0.0, max_attention=0.02):
    os.makedirs('./figures', exist_ok=True)
    os.makedirs(f'./figures/{filename}', exist_ok=True)
    os.makedirs(f'./figures/{filename}/attention_maps', exist_ok=True)
    os.makedirs(f'./figures/{filename}/attention_maps/{video_path}', exist_ok=True)
    os.makedirs(f'./figures/{filename}/attention_maps/{video_path}/{prompt}', exist_ok=True)
    avg_attention_maps = torch.cat(attentions[0], 0).mean(1).cpu().detach().numpy()
    avg_attention_maps = avg_attention_maps.clip(min_attention, max_attention)
    for layer_nr, attention_map in enumerate(avg_attention_maps):
        sns.heatmap(attention_map, cmap='viridis', cbar=True, vmin=min_attention, vmax=max_attention)
        plt.xlabel('Token index')
        plt.ylabel('Token index')
        plt.axvline(modal_token_position, color='red', linestyle='dashed', linewidth=1, label='begin_image_video')
        plt.axvline(modal_token_position + num_image_video_tokens, color='red', linestyle='dashed', linewidth=1,label='end_image_video')
        plt.axhline(modal_token_position, color='red', linestyle='dashed', linewidth=1)
        plt.axhline(modal_token_position + num_image_video_tokens, color='red', linestyle='dashed', linewidth=1)
        plt.title(f'Layer {layer_nr}')
        plt.savefig(f'./figures/{filename}/attention_maps/{video_path}/{prompt}/layer_{layer_nr}.png')
        plt.close()



def visualize_attention_vectors(attentions, output_ids, tokenizer, modal_token_position, num_image_video_tokens, filename, video_path, prompt, min_attention=0.0, max_attention=0.02):
    os.makedirs('./figures', exist_ok=True)
    os.makedirs(f'./figures/{filename}', exist_ok=True)
    os.makedirs(f'./figures/{filename}/attention_vectors', exist_ok=True)
    os.makedirs(f'./figures/{filename}/attention_vectors/{video_path}', exist_ok=True)
    os.makedirs(f'./figures/{filename}/attention_vectors/{video_path}/{prompt}', exist_ok=True)

    # Decode the output_ids to tokens
    output_ids = output_ids.squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(output_ids)
    tokens = [token.replace('▁', ' ') for token in tokens]

    # Distinguish repeated tokens
    for i in range(1, len(tokens)):
        for j in range(i):
            if tokens[i] == tokens[j]:
                tokens[i] = f'{tokens[i]}_{i}'

    start_image_video = modal_token_position + 1
    end_image_video = start_image_video + num_image_video_tokens
    end_input = attentions[0][0].shape[-1]
    #clip the attention values
    attention_vector = [torch.cat(attention, 0).mean(dim=(0, 1)).cpu().detach().numpy() for attention in attentions]
    attention_vector = [np.clip(attention, min_attention, max_attention) for attention in attention_vector]
    # save the attention vectors
    attention_vector_only_input = np.array([attention[0, :end_input] for attention in attention_vector[1:]])
    pickle.dump(attention_vector_only_input, open(f'./figures/{filename}/attention_vectors/{video_path}/{prompt}/attention_vector.pkl', 'wb'))
    for idx, attention in enumerate(tqdm(attention_vector)):
        fig, ax = plt.subplots(figsize=(20, 5))
        # Add vertical dashed lines to delimit the segments
        line_begin_image = ax.axvline(x=start_image_video - 0.5, color='red', linestyle='dashed', linewidth=1,
                                      label='begin_image_video')
        line_end_image = ax.axvline(x=end_image_video - 0.5, color='red', linestyle='dashed', linewidth=1,
                                    label='end_image_video')
        line_end_input = ax.axvline(x=end_input - 0.5, color='black', linestyle='dashed', linewidth=1,
                                    label='end_input')
        if idx == 0:
            sns.heatmap(attention, cmap='viridis', cbar=True, ax=ax, vmin=min_attention, vmax=max_attention)
            line_begin_image_ = ax.axhline(y=start_image_video - 0.5, color='red', linestyle='dashed', linewidth=1,
                                           label='begin_image_video')
            line_end_image_ = ax.axhline(y=end_image_video - 0.5, color='red', linestyle='dashed', linewidth=1,
                                         label='end_image_video')
            line_end_input_ = ax.axhline(y=end_input - 0.5, color='black', linestyle='dashed', linewidth=1,
                                         label='end_input')
            ax.legend(handles=[line_begin_image, line_end_image, line_end_input, line_begin_image_, line_end_image_,
                               line_end_input_], loc='upper center')
        else:
            # Plot the attention vector as a bar plot
            ax.bar(range(len(attention[0])), attention[0], color='blue')
            plt.ylim(min_attention, max_attention)
            ax.legend(handles=[line_begin_image, line_end_image, line_end_input], loc='upper center')

            ax.set_xticks(range(len(attention)))
            ax.set_xlabel('Previous Tokens')
            ax.set_ylabel('Attention')
        ax.set_title(f'Attention for Token {tokens[idx]}')
        plt.tight_layout()

        plt.savefig(f'./figures/{filename}/attention_vectors/{video_path}/{prompt}/token_{idx}.png')
        plt.close()


def compare_distributions(data, bins=50, hidden_states_or_attention='hidden_states'):
    num_data = len(data)
    results = {}
    os.makedirs('./figures', exist_ok=True)
    if hidden_states_or_attention == 'hidden_states':
        os.makedirs(f'./figures/distribution_comparison', exist_ok=True)
    else:
        os.makedirs(f'./figures/distribution_comparison/attention_vectors', exist_ok=True)
    for (i, j) in itertools.combinations(range(num_data), 2):
        data1 = data[i]
        data2 = data[j]
        prompt1 = data1['prompt']
        prompt2 = data2['prompt']
        video1 = data1['video']
        video2 = data2['video']
        data1 = data1['data']
        data2 = data2['data']
        # 1. Histogram Comparison
        plt.figure(figsize=(12, 6))
        plt.hist(data1, bins=bins, alpha=0.5, label=f'Video {video1}')
        plt.hist(data2, bins=bins, alpha=0.5, label=f'Video {video2}')
        plt.title(f'Histogram Comparison: video {video1}, prompt {prompt1} vs video {video2}, prompt {prompt2}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        if hidden_states_or_attention == 'hidden_states':
            plt.savefig(f'./figures/distribution_comparison/histogram_comparison_{video1}_{prompt1}_vs_{video2}_{prompt2}.png')
        else:
            plt.savefig(f'./figures/distribution_comparison/attention_vectors/histogram_comparison_{video1}_{prompt1}_vs_{video2}_{prompt2}.png')
        plt.close()

        # 2. Kolmogorov-Smirnov Test
        ks_stat, ks_p_value = ks_2samp(data1, data2)
        print(f"Kolmogorov-Smirnov Test: Video {video1}, Prompt {prompt1} vs Video {video2}, Prompt {prompt2}: Statistic={ks_stat}, p-value={ks_p_value}")

        # 4. KL Divergence
        # Create histogram-based probability distributions
        hist1, bin_edges1 = np.histogram(data1, bins=bins, density=True)
        hist2, bin_edges2 = np.histogram(data2, bins=bins, density=True)
        kl_divergence = entropy(hist1 + 1e-10, hist2 + 1e-10)  # Adding small value to avoid division by zero
        print(f"KL Divergence: Video {video1}, Prompt {prompt1} vs Video {video2}, Prompt {prompt2}: {kl_divergence}")

        if hidden_states_or_attention == 'hidden_states':
            with open(f'./figures/distribution_comparison/{video1}_{prompt1}_vs_{video2}_{prompt2}.txt', 'w') as f:
                f.write(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, p-value={ks_p_value}\n")
                f.write(f"KL Divergence: {kl_divergence}\n")
        else:
            with open(f'./figures/distribution_comparison/attention_vectors/{video1}_{prompt1}_vs_{video2}_{prompt2}.txt', 'w') as f:
                f.write(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, p-value={ks_p_value}\n")
                f.write(f"KL Divergence: {kl_divergence}\n")

        # 5. Cumulative Distribution Function (CDF) Plot
        cdf1 = np.cumsum(hist1 * np.diff(bin_edges1))
        cdf2 = np.cumsum(hist2 * np.diff(bin_edges2))

        plt.figure(figsize=(12, 6))
        plt.plot(bin_edges1[1:], cdf1, label=f'Video {video1} CDF')
        plt.plot(bin_edges2[1:], cdf2, label=f'Video {video2} CDF')
        plt.title(f'Cumulative Distribution Function (CDF) Comparison:  Video {video1}, Prompt {prompt1} vs Video {video2}, Prompt {prompt2}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Cumulative Probability')
        plt.legend(loc='upper left')
        if hidden_states_or_attention == 'hidden_states':
            plt.savefig(f'./figures/distribution_comparison/cdf_comparison_{video1}_{prompt1}_vs_{video2}_{prompt2}.png')
        else:
            plt.savefig(f'./figures/distribution_comparison/attention_vectors/cdf_comparison_{video1}_{prompt1}_vs_{video2}_{prompt2}.png')
        plt.close()


def plot_sim_and_tsne(metric: torch.Tensor, video_name:str, name: str, layer_num: int, frame_num: int):
    """
    Plot similarity matrix and t-SNE visualization.

    Args:
        metric (torch.Tensor): The metric to visualize, in shape [batch_size, num_tokens, token_size].
    """
    # Ensure the input tensor is on the GPU
    metric = metric.to('cuda')

    batch_size, num_tokens, token_size = metric.shape

    # Flatten the metric to shape [batch_size * num_tokens, token_size]
    flat_metric = metric.reshape(-1, token_size)

    # Normalize the flat_metric for cosine similarity calculation
    norms = torch.norm(flat_metric, dim=1, keepdim=True)
    normalized_metric = flat_metric / norms

    # Compute similarity matrix on the GPU
    similarity_matrix = torch.matmul(normalized_metric, normalized_metric.T)

    # Move the similarity matrix to CPU and convert to numpy for plotting
    similarity_matrix = similarity_matrix.cpu().numpy()

    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title(f"Token Similarity Matrix {name}")
    plt.xlabel("Tokens (Flattened Batches)")
    plt.ylabel("Tokens (Flattened Batches)")

    save_dir = f"./figures/sys_user_prompt_comp/{video_name}/layer_{layer_num}/frame_{frame_num}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"similarity_matrix_{name}.png"))

    # Compute t-SNE visualization
    tsne = TSNE(n_components=2, random_state=0)
    # Move the metric back to CPU and convert to numpy for t-SNE
    flat_metric = flat_metric.cpu().numpy()
    # Standardizing the data before applying t-SNE
    standardized_data = StandardScaler().fit_transform(flat_metric)
    tsne_results = tsne.fit_transform(standardized_data)

    # Prepare t-SNE plot
    plt.figure(figsize=(10, 8))
    s = range(20, 0, -5)
    for batch in range(batch_size):
        indices = list(range(batch * num_tokens, (batch + 1) * num_tokens))
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Batch {batch}', s=s[batch])  # Smaller dots

    plt.title(f"t-SNE of Tokens {name}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"tsne_{name}.png"))


