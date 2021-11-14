import argparse
import pickle

import numpy as np
import torch
import clip
from transformers import BertTokenizer, BertModel

TEXT_PATHS = {
    "train": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/train/text/en/raw_text/train.text.id.en",
    "val": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/val/text/en/raw_text/val.text.id.en",
    "test": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/test/text/en/raw_text/test.text.id.en"
}


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_text(key, ids):
    file_path = TEXT_PATHS[key]
    dict_text = {}
    with open(file_path) as fp:
        for line in fp:
            id, text = line.split(" ", 1)  # first space separates id from text
            if id in ids:
                dict_text[id] = text
    sentence_list = [v for _, v in sorted(dict_text.items())]  # it's important that the result is sorted by clip ID
    print(f"len(sentence_list): {len(sentence_list)}", flush=True)
    return sentence_list


# obtain embeddings for each sentence in the input list
def obtain_embeddings(key, ids, method="BERT"):
    sentence_list = load_text(key, ids)
    
    print(f"sentence_list.shape {sentence_list.shape}",  flush=True)

    if method=="clip":
        model, _ = clip.load('ViT-B/32', device)        
        sentence_tensor = torch.cat([clip.tokenize(sent, truncate=True) for sent in sentence_list]).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(sentence_tensor)
        return embeddings.cpu().numpy()

    if method=="BERT":
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print(sentence_list[:4], flush=True)
        # idxs_segmIDs = tokenizer.tokenize(sentence_list, add_special_tokens=True, padding="max_length",
        #                                   max_length=32, truncation=True, return_tensors="pt")
        idxs_segmIDs = tokenizer.batch_encode_plus(sentence_list, add_special_tokens=True, padding="max_length",
                                                   max_length=32, truncation=True, return_tensors="pt")

        print(f"type(idxs_degmIDs) {type(idxs_segmIDs)}", flush=True)

        indexed_tokens = idxs_segmIDs["input_ids"]
        segments_ids = idxs_segmIDs["token_type_ids"]
        attention_mask = idxs_segmIDs["attention_mask"]

        print(f"segments_ids.shape {segments_ids.shape}", flush=True)
        print(f"attention_mask.shape {attention_mask.shape}", flush=True)

        model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states=True)
        model.eval()
        with torch.no_grad():
            outputs = model(indexed_tokens, attention_mask)
            print(f"type(outputs) {type(outputs)}", flush=True)
            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # becase we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        print(type(hidden_states), flush=True)
        for hidden_state in hidden_states:
            print("*********************************+", flush=True)
            print('Tensor shape for each layer: ', hidden_state[0].shape, flush=True)

        # # Stores the token vectors, with shape [22 x 768]
        # token_vecs_sum = []

        # # `token_embeddings` is a [22 x 12 x 768] tensor.

        # # For each token in the sentence...
        # for token in token_embeddings:

        #     # `token` is a [12 x 768] tensor

        #     # Sum the vectors from the last four layers.
        #     sum_vec = torch.sum(token[-4:], dim=0)

        #     # Use `sum_vec` to represent `token`.
        #     token_vecs_sum.append(sum_vec)

        # print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


# returns the ID of those clips for which text is available
def get_clip_ids(key):
    file_path = TEXT_PATHS[key]
    id_list = []
    with open(file_path) as fp:
        for line in fp:
            id, text = line.split(" ", 1)  # first space separates id from text
            id_list.append(id)
    return id_list


def process_text(subset=0.005):
    for key in TEXT_PATHS:
        sentence_list = load_text(key)
        idx_max = int(len(sentence_list)*subset)
        print(f"idx_max: {idx_max}", flush=True)
        sentence_list = sentence_list[0:idx_max]
        sentence_tensor = torch.cat([clip.tokenize(sent, truncate=True) for sent in sentence_list]).to(device)

        print(sentence_tensor.shape, flush=True)
        embeddings = obtain_embeddings(sentence_tensor)
        print(embeddings.shape, flush=True)
        # #Store sentences & embeddings on disk
        with open(f'video_data/{key}_sentence_embeddings.pkl', "wb") as fOut:
            pickle.dump(embeddings.numpy(), fOut, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'video_data/{key}_sentence_raw.pkl', "wb") as fOut:
            pickle.dump(sentence_list, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved sentence embeddings.", flush=True)


def obtain_avg_embed(key="train", subset=1):
    clip_ids_text = get_clip_ids(key=key)
    ids = sorted(clip_ids_text)
    idx_max = int(len(ids)*subset)
    embeds = obtain_embeddings(key, ids[0:idx_max])  # obtain text embeddings for each clip
    col_mean = np.average(embeds, axis=0)
    return np.tile(col_mean, (embeds.shape[0],1))


if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file_path', type=str, default="/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/test/text/en/raw_text/test.text.id.en", help="path to the file where text dataset is located")
    # args = parser.parse_args()
    pass
