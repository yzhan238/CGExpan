import numpy as np
from tqdm import tqdm
import json
import mmap
import copy

MAX_SENT_LEN = 300


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def load_vocab(filename):
    eid2name = {}
    keywords = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[1])
            eid2name[eid] = temp[0]
            keywords.append(eid)
    eid2idx = {w:i for i, w in enumerate(keywords)}
    print(f'Vocabulary: {len(keywords)} keywords loaded')
    return eid2name, keywords, eid2idx


def get_masked_sentences(filename, mask_token, eid2idx):
    sentences = []
    total_line = get_num_lines(filename)
    entity_num = [0 for _ in eid2idx]
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_line):
            obj = json.loads(line)
            if len(obj['entityMentions']) == 0 or len(obj['tokens']) > MAX_SENT_LEN:
                continue
            raw_sent = [token.lower() for token in obj['tokens']]
            for entity in obj['entityMentions']:
                eid = entity['entityId']
                if eid not in eid2idx:
                    continue
                entity_num[eid2idx[eid]] += 1
                sent = copy.deepcopy(raw_sent)
                sent[entity['start']:entity['end']+1] = [mask_token]
                sentences.append((eid, sent))
    print(f'Sentences: {len(sentences)} masked sentences constructed')
    entity_pos = np.cumsum(entity_num)
    entity_pos = [0] + list(entity_pos)
    return sentences, entity_pos


def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)