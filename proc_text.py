import re
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


def natural_keys(text):
    def atof(text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def _groupByClip(dict_text):

    ####
    # vid_feats_files = glob.glob(f"{data_dir}/{key}_vid_feats_*.pkl")
    # vid_feats_files.sort(key=natural_keys)
    ####

    utterance_ids = list(dict_text.keys()).sort(key=natural_keys)

    print(len(utterance_ids), flush=True)
    print(utterance_ids, flush=True)
    
    dict_text_grouped = {}
    clip_id = utterance_ids[0][:11]
    dict_text_grouped[clip_id] = dict_text[utterance_ids[0]]
    
    
    for utt_id in utterance_ids:
        if utt_id[:11] not in dict_text_grouped:
            dict_text_grouped[utt_id[:11]] = dict_text[utt_id]
        else:
            dict_text_grouped[utt_id[:11]] += dict_text[utt_id]

    print(dict_text_grouped, flush=True)
    raise Exception("quiet aquí!!")
    return dict_text_grouped


def load_text(key, ids, groupByClip=False):
    file_path = TEXT_PATHS[key]
    dict_text = {}
    with open(file_path) as fp:
        for line in fp:
            id, text = line.split(" ", 1)  # first space separates id from text
            if id in ids:
                dict_text[id] = text

    print(dict_text.keys(), flush=True)

    if groupByClip:
        dict_text = _groupByClip(dict_text)
    else:
        sentence_list = [v for _, v in sorted(dict_text.items())]  # it's important that the result is sorted by clip ID
    print(f"len(sentence_list): {len(sentence_list)}", flush=True)
    return sentence_list


# obtain embeddings for each sentence in the input list
def obtain_embeddings(key, ids, method="BERT", groupByClip=False):
    sentence_list = load_text(key, ids, groupByClip=groupByClip)
    
    print(f"len(sentence_list) {len(sentence_list)}",  flush=True)

    if method=="clip":
        model, _ = clip.load('ViT-B/32', device)        
        sentence_tensor = torch.cat([clip.tokenize(sent, truncate=True) for sent in sentence_list]).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(sentence_tensor)
        return embeddings.cpu().numpy()

    if method=="BERT":
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        idxs_segmIDs = tokenizer.batch_encode_plus(sentence_list, add_special_tokens=True, padding="max_length",
                                                   max_length=32, truncation=True, return_tensors="pt")

        indexed_tokens = idxs_segmIDs["input_ids"]
        segments_ids = idxs_segmIDs["token_type_ids"]
        attention_mask = idxs_segmIDs["attention_mask"]

        model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states=True)
        model.eval()
        with torch.no_grad():
            outputs = model(indexed_tokens, attention_mask)
            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # becase we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]
            # hidden_states contains 12 hidden states of shape Bx32x768 (32 tokes per sentence)

        hidden_states = torch.sum(torch.stack(hidden_states[-4:], dim=0), dim=0)  # Sum the vectors from the last four layers.
        print(f"hidden_states.shape {hidden_states.shape}", flush=True)
        return hidden_states  # shape Bx32x768


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
