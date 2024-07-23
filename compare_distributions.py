import pickle

from videollama2.mm_utils import compare_distributions

"""
# load distributions
pickle_vicuna_what = open("./figures/vicuna7b/hidden_states/whiskers/What/similarity_matrix.pkl", "rb")
pickle_vicuna_summarize = open("./figures/vicuna7b/hidden_states/whiskers/Summarize/similarity_matrix.pkl", "rb")
pickle_mistral_cat_chicken_what = open("./figures/mistral/hidden_states/cat_and_chicken/What/similarity_matrix.pkl", "rb")
pickle_mistral_cat_chicken_summarize = open("./figures/mistral/hidden_states/cat_and_chicken/Summarize/similarity_matrix.pkl", "rb")
pickle_mistral_road_accident_what = open("./figures/mistral/hidden_states/RoadAccidents127_x264/What/similarity_matrix.pkl", "rb")
pickle_mistral_road_accident_summarize = open("./figures/mistral/hidden_states/RoadAccidents127_x264/Summarize/similarity_matrix.pkl", "rb")
pickle_mistral_solid_black_what = open("./figures/mistral/hidden_states/solid_black/What/similarity_matrix.pkl", "rb")
pickle_mistral_solid_black_summarize = open("./figures/mistral/hidden_states/solid_black/Summarize/similarity_matrix.pkl", "rb")

vicuna_what = pickle.load(pickle_vicuna_what)
vicuna_what = vicuna_what.flatten()
vicuna_what = {
    "video": "whiskers",
    "prompt": "What",
    "data": vicuna_what
}
vicuna_summarize = pickle.load(pickle_vicuna_summarize)

vicuna_summarize = vicuna_summarize.flatten()
vicuna_summarize = {
    "video": "whiskers",
    "prompt": "Summarize",
    "data": vicuna_summarize
}

mistral_cat_chicken_what = pickle.load(pickle_mistral_cat_chicken_what)
mistral_cat_chicken_what = mistral_cat_chicken_what.flatten()
mistral_cat_chicken_what = {
    "video": "cat_and_chicken",
    "prompt": "What",
    "data": mistral_cat_chicken_what
}
mistral_cat_chicken_summarize = pickle.load(pickle_mistral_cat_chicken_summarize)
mistral_cat_chicken_summarize = mistral_cat_chicken_summarize.flatten()
mistral_cat_chicken_summarize = {
    "video": "cat_and_chicken",
    "prompt": "Summarize",
    "data": mistral_cat_chicken_summarize
}

mistral_road_accident_what = pickle.load(pickle_mistral_road_accident_what)
mistral_road_accident_what = mistral_road_accident_what.flatten()
mistral_road_accident_what = {
    "video": "RoadAccidents127_x264",
    "prompt": "What",
    "data": mistral_road_accident_what
}

mistral_road_accident_summarize = pickle.load(pickle_mistral_road_accident_summarize)
mistral_road_accident_summarize = mistral_road_accident_summarize.flatten()
mistral_road_accident_summarize = {
    "video": "RoadAccidents127_x264",
    "prompt": "Summarize",
    "data": mistral_road_accident_summarize
}

mistral_solid_black_what = pickle.load(pickle_mistral_solid_black_what)
mistral_solid_black_what = mistral_solid_black_what.flatten()
mistral_solid_black_what = {
    "video": "solid_black",
    "prompt": "What",
    "data": mistral_solid_black_what
}

mistral_solid_black_summarize = pickle.load(pickle_mistral_solid_black_summarize)
mistral_solid_black_summarize = mistral_solid_black_summarize.flatten()
mistral_solid_black_summarize = {
    "video": "solid_black",
    "prompt": "Summarize",
    "data": mistral_solid_black_summarize
}

data = [vicuna_what, vicuna_summarize, mistral_cat_chicken_what, mistral_cat_chicken_summarize, mistral_road_accident_what, mistral_road_accident_summarize, mistral_solid_black_what, mistral_solid_black_summarize]

# compare distributions
compare_distributions(data)

"""
#load attention vectors
pickle_vicuna_what = open("./figures/vicuna7b/attention_vectors/whiskers/What/attention_vector.pkl", "rb")
#pickle_vicuna_summarize = open("./figures/vicuna7b/attention_vectors/whiskers/Summarize/attention_vector.pkl")
pickle_mistral_cat_chicken_what = open("./figures/mistral/attention_vectors/cat_and_chicken/What/attention_vector.pkl", "rb")
pickle_mistral_cat_chicken_summarize = open("./figures/mistral/attention_vectors/cat_and_chicken/Summarize/attention_vector.pkl", "rb")
pickle_mistral_road_accident_what = open("./figures/mistral/attention_vectors/RoadAccidents127_x264/What/attention_vector.pkl", "rb")
pickle_mistral_road_accident_summarize = open("./figures/mistral/attention_vectors/RoadAccidents127_x264/Summarize/attention_vector.pkl", "rb")
pickle_mistral_solid_black_what = open("./figures/mistral/attention_vectors/solid_black/What/attention_vector.pkl", "rb")
pickle_mistral_solid_black_summarize = open("./figures/mistral/attention_vectors/solid_black/Summarize/attention_vector.pkl", "rb")

vicuna_what = pickle.load(pickle_vicuna_what)
vicuna_what = vicuna_what.flatten()
vicuna_what = {
    "video": "whiskers",
    "prompt": "What",
    "data": vicuna_what
}
"""vicuna_summarize = pickle.load(pickle_vicuna_summarize)
vicuna_summarize = {
    "video": "whiskers",
    "prompt": "Summarize",
    "data": vicuna_summarize
}
"""
mistral_cat_chicken_what = pickle.load(pickle_mistral_cat_chicken_what)
mistral_cat_chicken_what = mistral_cat_chicken_what.flatten()
mistral_cat_chicken_what = {
    "video": "cat_and_chicken",
    "prompt": "What",
    "data": mistral_cat_chicken_what
}
mistral_cat_chicken_summarize = pickle.load(pickle_mistral_cat_chicken_summarize)
mistral_cat_chicken_summarize = mistral_cat_chicken_summarize.flatten()
mistral_cat_chicken_summarize = {
    "video": "cat_and_chicken",
    "prompt": "Summarize",
    "data": mistral_cat_chicken_summarize
}

mistral_road_accident_what = pickle.load(pickle_mistral_road_accident_what)
mistral_road_accident_what = mistral_road_accident_what.flatten()
mistral_road_accident_what = {
    "video": "RoadAccidents127_x264",
    "prompt": "What",
    "data": mistral_road_accident_what
}

mistral_road_accident_summarize = pickle.load(pickle_mistral_road_accident_summarize)
mistral_road_accident_summarize = mistral_road_accident_summarize.flatten()
mistral_road_accident_summarize = {
    "video": "RoadAccidents127_x264",
    "prompt": "Summarize",
    "data": mistral_road_accident_summarize
}

mistral_solid_black_what = pickle.load(pickle_mistral_solid_black_what)
mistral_solid_black_what = mistral_solid_black_what.flatten()
mistral_solid_black_what = {
    "video": "solid_black",
    "prompt": "What",
    "data": mistral_solid_black_what
}

mistral_solid_black_summarize = pickle.load(pickle_mistral_solid_black_summarize)
mistral_solid_black_summarize = mistral_solid_black_summarize.flatten()
mistral_solid_black_summarize = {
    "video": "solid_black",
    "prompt": "Summarize",
    "data": mistral_solid_black_summarize
}


data = [vicuna_what, mistral_cat_chicken_what, mistral_cat_chicken_summarize, mistral_road_accident_what, mistral_road_accident_summarize, mistral_solid_black_what, mistral_solid_black_summarize]

# compare attention vectors

compare_distributions(data, hidden_states_or_attention="attention_vectors")