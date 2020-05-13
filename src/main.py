import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import argparse
from utils import *
from CGExpan import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-sent', default='sentences.json', help='sent file')
    parser.add_argument('-emb_file', default='pretrained_emb.npy', help='name of pretrained embedding npy file')
    parser.add_argument('-entity_pos_out', default='entity_pos.pkl', help='name of entity index file')
    parser.add_argument('-output', default='results', help='file name for output')
    parser.add_argument('-k', default=5, help='k for soft match', type=int)
    parser.add_argument('-m', default=2, help='margin', type=int)
    parser.add_argument('-gen_thres', default=3, help='class name threshold', type=int)
    args = parser.parse_args()

    cgexpan = CGExpan(args, torch.device("cuda:0"))

    print(args)

    if not os.path.exists(os.path.join(args.dataset, args.output)):
        os.mkdir(os.path.join(args.dataset, args.output))

    for file in os.listdir(os.path.join(args.dataset, 'query')):
        query_sets = []
        with open(os.path.join(args.dataset, 'query', file), encoding='utf-8') as f:
            for line in f:
                if line == 'EXIT\n': break
                temp = line.strip().split(' ')
                query_sets.append([int(eid) for eid in temp])
        gt = set()
        with open(os.path.join(args.dataset, 'gt', file), encoding='utf-8') as f:
            for line in f:
                temp = line.strip().split('\t')
                eid = int(temp[0])
                if int(temp[2]) >= 1:
                    gt.add(eid)

        for i, query_set in enumerate(query_sets):
            expanded = cgexpan.expand(query_set, 50, args.m, gt)
            with open(os.path.join(args.dataset, args.output, f'{i}_{file}'), 'w') as f:
                print(apk(gt, expanded, 10), file=f)
                print(apk(gt, expanded, 20), file=f)
                print(apk(gt, expanded, 50), file=f)
                print('', file=f)
                for eid in expanded:
                    print(f'{eid}\t{cgexpan.eid2name[eid]}', file=f)