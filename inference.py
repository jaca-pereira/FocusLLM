import argparse

import torch
import transformers

import sys
sys.path.append('./')
from videollama2.conversation import conv_templates
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image, visualize_average_attention, visualize_attention_vectors, \
    visualize_hidden_states, visualize_hidden_states_distribution
from videollama2.model.builder import load_pretrained_model


def inference():
    # Video Inference
    paths = ['assets/cat_and_chicken.mp4']
    #questions = ['Summarize the events in the video and name the main animals that appear.'] #para replicar o link
    #questions = ['Summarize the events in the video and name the main objects that appear.'] #QUANDO PEDIMOS OBJETOS ELE COMPORTA-SE DE FORMA ESTRANHA. Ou ent quando é a dividir por 4  e a mask nao fica bem setup ele começa a dar links
    questions = ['Describe the video.']
    #questions = ['What is your opinion on the goal scored by Cristiano Ronaldo?']
    modal_list = ['video']
    #modal_list = ['image']

    # 1. Initialize the model.
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
    # Base model inference (only need to replace model_path)
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-Base'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.to('cuda:0')
    model.get_model().config.pad_token = tokenizer.pad_token_id
    model.get_model().config.ratio = 2  # max 2
    #model.get_model().config.merge_layer = 27 #para replicar os links, foi feito com o merge nos tokens todos, agr o merge esta so nos de texto
    model.get_model().config.merge_layer = 26  #TEORIA: DEVIAMOS LIMITAR O MERGE PARA APENAS OS TOKENS QUE NÃO SÃO RELEVANTES PARA NXTP DO PRIMEIRO TOKEN
    conv_mode = 'llama_2'

    # 2. Visual preprocess (load & transform image or video).
    if modal_list[0] == 'video':
        tensor = process_video(paths[0], processor, model.config.image_aspect_ratio).to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    else:
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
        modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
    tensor = [tensor]

    # 3. text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda:0')

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    modal_token_position = (input_ids == modal_token_index).nonzero()[0, 1].item()
    output_ids = outputs.sequences
    """attentions = outputs.attentions
    hidden_states = outputs.hidden_states
    prompt = questions[0].split(" ")[0]
    video_path = paths[0].split('/')[-1].removesuffix('.mp4')
    filename = "mistral"
    #visualize_hidden_states(hidden_states, modal_token_position, model.model.image_video_tokens, filename, video_path, prompt)
    #visualize_hidden_states_distribution(hidden_states, filename, video_path, prompt)
    #visualize_average_attention(attentions, modal_token_position, model.model.image_video_tokens, filename, video_path, prompt)
    visualize_attention_vectors(attentions, output_ids, tokenizer, modal_token_position, model.model.image_video_tokens, filename, video_path, prompt)"""
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(response[0])


if __name__ == "__main__":
    inference()