import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import argparse
import pickle
from utils import *


def get_pretrained_emb(model, tokenizer, sentences, entity_pos, eid2idx, np_file, dim=768, batch_size=64):
    fp = np.memmap(np_file, dtype='float32', mode='w+', shape=(entity_pos[-1], dim))
    ptr_list = [0 for _ in entity_pos[:-1]]
    iterations = int(len(sentences)/batch_size) + (0 if len(sentences) % batch_size == 0 else 1)
    for i in tqdm(range(iterations)):
        start = i * batch_size
        end = min((i+1)*batch_size, len(sentences))
        batch_ids = []
        for _, sent in sentences[start:end]:
            ids = tokenizer.encode(sent, max_length=512)
            batch_ids.append(ids)
        batch_max_length = max(len(ids) for ids in batch_ids)
        ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()
        masks = (ids != 0).long()
        temp = (ids == tokenizer.mask_token_id).nonzero()
        mask_pos = []
        for ti, t in enumerate(temp):
            assert t[0].item() == ti
            mask_pos.append(t[1].item())
        ids = ids.to('cuda')
        masks = masks.to('cuda')
        with torch.no_grad():
            batch_final_layer = model(ids, masks)[0]
        for final_layer, mask, (eid, _) in zip(batch_final_layer, mask_pos, sentences[start:end]):
            rep = final_layer[mask].cpu().numpy()
            this_idx = entity_pos[eid2idx[eid]] + ptr_list[eid2idx[eid]]
            ptr_list[eid2idx[eid]] += 1
            fp[this_idx] = rep.astype(np.float32)
    del fp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-sent', default='sentences.json', help='sent file')
    parser.add_argument('-npy_out', default='pretrained_emb.npy', help='name of output npy file')
    parser.add_argument('-entity_pos_out', default='entity_pos.pkl', help='name of output entity index file')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print('CUDA not available')
        exit()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = False)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()

    _, _, eid2idx = load_vocab(os.path.join(args.dataset, args.vocab))
    sentences, entity_pos = get_masked_sentences(os.path.join(args.dataset, args.sent), tokenizer.mask_token, eid2idx)

    pickle.dump(entity_pos, open(os.path.join(args.dataset, args.entity_pos_out), 'wb'))
    get_pretrained_emb(model, tokenizer, sentences, entity_pos, eid2idx, np_file=os.path.join(args.dataset, args.npy_out))