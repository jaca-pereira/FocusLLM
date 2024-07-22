#from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, BertTokenizer
import sys

gettrace = getattr(sys, 'gettrace', None)

if gettrace is None:

    print('No sys.gettrace')

elif gettrace():

    import pydevd_pycharm

    pydevd_pycharm.settrace('localhost', port=9000, stdoutToServer=True, stderrToServer=True)


#text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")
#text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
print("CLIP Text FINISHED")

#image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
#cfg_only = CLIPVisionConfig.from_pretrained('openai/clip-vit-large-patch14-336')
#vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')
print("CLIP Vision FINISHED")

#qformer_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("QFORMER FINISHED")

print("FINISHED")