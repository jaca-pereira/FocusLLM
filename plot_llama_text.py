import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from videollama2.mm_utils import visualize_hidden_states, visualize_attention_vectors, visualize_average_attention, \
    visualize_hidden_states_distribution

# Load the model and tokenizer
model_name = 'vicuna7b'
tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained('lmsys/vicuna-7b-v1.5', torch_dtype=torch.float16, device_map="auto")


# Define the input text
input_text = """Whiskers the cat was a special cat. Whiskers the cat was a special cat. Whiskers the cat was a special cat. Whiskers loved being a cat. Whiskers loved being a cat. Whiskers loved being a cat. Whiskers had the softest fur, the softest fur a cat could have. Whiskers had the softest fur, the softest fur a cat could have. Whiskers had the softest fur, the softest fur a cat could have. Whiskers had the longest whiskers, the longest whiskers any cat could have. Whiskers had the longest whiskers, the longest whiskers any cat could have. Whiskers had the longest whiskers, the longest whiskers any cat could have. Whiskers' whiskers were his pride, his pride as a cat. Whiskers' whiskers were his pride, his pride as a cat. Whiskers' whiskers were his pride, his pride as a cat.

Every morning, Whiskers would wake up as a cat, stretch as a cat, and yawn as a cat. Every morning, Whiskers would wake up as a cat, stretch as a cat, and yawn as a cat. Every morning, Whiskers would wake up as a cat, stretch as a cat, and yawn as a cat. Whiskers loved to stretch and yawn like any cat. Whiskers loved to stretch and yawn like any cat. Whiskers loved to stretch and yawn like any cat. After stretching and yawning, Whiskers would go to the kitchen, the kitchen where he ate, where he ate like a cat. After stretching and yawning, Whiskers would go to the kitchen, the kitchen where he ate, where he ate like a cat. After stretching and yawning, Whiskers would go to the kitchen, the kitchen where he ate, where he ate like a cat. Whiskers loved eating his breakfast, his breakfast as a cat. Whiskers loved eating his breakfast, his breakfast as a cat. Whiskers loved eating his breakfast, his breakfast as a cat.

Whiskers had a favorite spot, a favorite spot where he would nap, where he would nap like a cat. Whiskers had a favorite spot, a favorite spot where he would nap, where he would nap like a cat. Whiskers had a favorite spot, a favorite spot where he would nap, where he would nap like a cat. This spot was on the windowsill, the windowsill where the sun shone, where the sun shone on Whiskers. This spot was on the windowsill, the windowsill where the sun shone, where the sun shone on Whiskers. This spot was on the windowsill, the windowsill where the sun shone, where the sun shone on Whiskers. Whiskers loved the sun, the sun that warmed his fur, his fur as a cat. Whiskers loved the sun, the sun that warmed his fur, his fur as a cat. Whiskers loved the sun, the sun that warmed his fur, his fur as a cat. Whiskers could nap for hours, nap for hours as only a cat could. Whiskers could nap for hours, nap for hours as only a cat could. Whiskers could nap for hours, nap for hours as only a cat could.

Whiskers enjoyed playing, playing with his toys, his toys as a cat. Whiskers enjoyed playing, playing with his toys, his toys as a cat. Whiskers enjoyed playing, playing with his toys, his toys as a cat. Whiskers had a favorite toy, a toy mouse, a toy mouse he loved as a cat. Whiskers had a favorite toy, a toy mouse, a toy mouse he loved as a cat. Whiskers had a favorite toy, a toy mouse, a toy mouse he loved as a cat. Whiskers would chase the toy mouse, chase the toy mouse around the house, around the house like a cat. Whiskers would chase the toy mouse, chase the toy mouse around the house, around the house like a cat. Whiskers would chase the toy mouse, chase the toy mouse around the house, around the house like a cat. Whiskers never got tired, never got tired of playing, playing with his toy mouse. Whiskers never got tired, never got tired of playing, playing with his toy mouse. Whiskers never got tired, never got tired of playing, playing with his toy mouse.

Whiskers also loved to watch birds, watch birds from the window, the window where he napped. Whiskers also loved to watch birds, watch birds from the window, the window where he napped. Whiskers also loved to watch birds, watch birds from the window, the window where he napped. Whiskers would sit and watch, sit and watch the birds, the birds outside. Whiskers would sit and watch, sit and watch the birds, the birds outside. Whiskers would sit and watch, sit and watch the birds, the birds outside. Whiskers dreamed of catching birds, dreamed like a cat dreams. Whiskers dreamed of catching birds, dreamed like a cat dreams. Whiskers dreamed of catching birds, dreamed like a cat dreams. Whiskers' whiskers would twitch, his whiskers would twitch as he watched the birds. Whiskers' whiskers would twitch, his whiskers would twitch as he watched the birds. Whiskers' whiskers would twitch, his whiskers would twitch as he watched the birds.
"""

prompt = f"""Summarize the following text: 
====================
{input_text}
====================
Summary:
"""
# Tokenize the input
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

# Generate text with attention scores
outputs = model.generate(
    input_ids,
    max_new_tokens=350,
    output_attentions=True,
    output_hidden_states=True,
    return_dict_in_generate=True
)

# Extract attention scores and generated tokens
attentions = outputs.attentions  # List of attentions from each layer
generated_ids = outputs.sequences
hidden_states = outputs.hidden_states
# Decode generated tokens
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated Text: {generated_text}")

# Plot the attention scores
visualize_hidden_states(hidden_states,-1, len(generated_ids[0]), model_name, "whiskers", "Summarize")
visualize_hidden_states_distribution(hidden_states, model_name, "whiskers", "Summarize")
visualize_average_attention(attentions, -1, len(input_ids[0]), model_name, "whiskers", "Summarize")
visualize_attention_vectors(attentions, generated_ids, tokenizer, -1, len(input_ids[0]), model_name, "whiskers", "Summarize")
