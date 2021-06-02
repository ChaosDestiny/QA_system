import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import pandas as pd
from heapq import nlargest

tokenizer = AutoTokenizer.from_pretrained('FPTAI/vibert-base-cased', do_lower_case=False)
model = AutoModel.from_pretrained('FPTAI/vibert-base-cased')
model.load_state_dict(torch.load('save/fpt_bert_model.pt', map_location=torch.device('cpu')))
data_df = pd.read_csv('save/qa_data.csv', header=None)
ans = data_df[1].tolist()
ques = data_df[0].tolist()
emb_str = data_df[2].tolist()
emb = [torch.from_numpy(np.fromstring(emb_str[i][2:-2], dtype=float, sep=",")).unsqueeze(0) for i in range(len(emb_str))]

del data_df, emb_str


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


def answer(txt):
    embedding = get_embed(txt)
    sim_score = [get_similarity(embedding, emb[i]) for i in range(1, len(emb))]
    # ans_id = sim_score.index(max(sim_score))
    ans_id = nlargest(3, range(len(sim_score)), key=lambda idx: sim_score[idx])
    del sim_score, embedding
    sim_q_list = [ques[ans_id[0]], ques[ans_id[1]], ques[ans_id[2]]]
    ans_list = [ans[ans_id[0]], ans[ans_id[1]], ans[ans_id[2]]]
    return sim_q_list, ans_list


# if __name__ == '__main__':
    # a = time.time()
    # print(answer('Cà tím có giúp giảm béo không?'))
    # b = time.time()
    # print(b-a)
