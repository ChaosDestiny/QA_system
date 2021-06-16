import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
import torch.nn.functional as F
import pandas as pd
import time 
from heapq import nlargest
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
# model = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('FPTAI/vibert-base-cased', do_lower_case=False)
model = AutoModel.from_pretrained('FPTAI/vibert-base-cased').to(device)

model.load_state_dict(torch.load('save/fpt_bert_model_v2.pt', map_location=device))
data_df = pd.read_csv('save/final_data_embedded_fpt.csv', header=None)
ids = data_df[0].tolist()
ques = data_df[1].tolist()
emb_str = data_df[2].tolist()
emb = [torch.from_numpy(np.fromstring(emb_str[i][2:-2], dtype=float, sep=",")).unsqueeze(0) for i in range(len(emb_str))]

del data_df, emb_str
# rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')


# def segment(txt):
#     ori_sent = []
#     sentences = rdrsegmenter.tokenize(txt)
#     for sentence in sentences:
#         ori_sent.append(' '.join(sentence))
#     txt = " ".join(ori_sent)
#     return txt


def get_embed(txt):
    for i in range(len(txt)):
        txt[i].lower()

    encoding = tokenizer(txt, max_length=128, truncation=True, padding='max_length', return_token_type_ids=False)

    encoding['input_ids'] = torch.from_numpy(np.array([encoding['input_ids']]))
    encoding['attention_mask'] = torch.from_numpy(np.array([encoding['attention_mask']]))

    # embed1 = model(encoding1['input_ids'], encoding1['attention_mask'])[0][:, 0]
    # embed2 = model(encoding2['input_ids'], encoding2['attention_mask'])[0][:, 0]

    sequence_output = model(encoding['input_ids'], encoding['attention_mask'])[0].unsqueeze(1)
    embed = F.avg_pool2d(sequence_output, (sequence_output.shape[2], 1)).squeeze(1).squeeze(1)

    return embed


def get_similarity(embed1, embed2):
    # embed1 = torch.from_numpy(np.array(list1))
    # embed2 = torch.from_numpy(np.array(list2))
    score = nn.functional.cosine_similarity(embed1, embed2).tolist()[0]
    return score


def answer(txt, num_id):
    embedding = get_embed(txt)
    sim_score = [get_similarity(embedding, emb[i]) for i in range(0, len(emb))]
    # ans_id = sim_score.index(max(sim_score))
    id_index = nlargest(num_id, range(len(sim_score)), key=lambda idx: sim_score[idx])
    del embedding
    sim_q_list = [ques[id_index[i]] for i in range(len(id_index))]
    id_list = [ids[id_index[i]] for i in range(len(id_index))]
    score = [sim_score[i] for i in id_index]
    return sim_q_list, id_list, score


# if __name__ == '__main__':
    # a = time.time()
    # print(answer('Cà tím có giúp giảm béo không?'))
    # b = time.time()
    # print(b-a)
    # print(segment('Cà tím có giúp giảm béo không?'))
    # print(os.system("java -version"))
