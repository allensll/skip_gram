import os
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)-1

    def __len__(self):
        return len(self.idx2word)


class UnigramTable(object):
    def __init__(self, data, n_tokens, power=0.75, tsz=int(1e6)):
        # print('building unigramtable ...')
        self.n_tokens = n_tokens
        self.power = 0.75
        self.tsz = tsz

        unigramtable = np.zeros(tsz)
        prob = np.zeros(n_tokens)
        for w in data:
            prob[w.item()] += 1
        prob /= len(data)
        sum_prob = sum(np.power(prob, power))
        prob = np.power(prob, power) / sum_prob
        idx = 0
        for i in range(n_tokens):
            num = int(prob[i] * tsz)
            unigramtable[idx:idx+num] = i
            idx += num
            # unigramtable = np.concatenate((unigramtable, [i] * num))
        assert len(unigramtable) <= tsz
        padding = np.random.randint(0, n_tokens, tsz - idx)
        unigramtable[idx:] = padding
        self.unigramtable = unigramtable

    def sample(self, n_sample, mask=-1):
        # 5 - 20, 2 - 5
        idx = np.random.randint(0, self.tsz, n_sample)
        res = self.unigramtable[idx]
        if mask == -1:
            return res
        else:
            for i in range(n_sample):
                if res[i] == mask:
                    while True:
                        val = self.unigramtable[np.random.randint(self.tsz)]
                        if not val == mask:
                            break
                    res[i] = val
            return res


class TextData(data.Dataset):
    def __init__(self, path):
        super(TextData, self).__init__()

        self.dictionary = Dictionary()
        self.train = self.tokenizer(os.path.join(path, 'train.txt'), train=True)
        # self.valid = self.tokenizer(os.path.join(path, 'valid.txt'))
        # self.test = self.tokenizer(os.path.join(path, 'test.txt'))
        self.n_tokens = len(self.dictionary)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def tokenizer(self, path, train=False):
        assert os.path.exists(path)

        with open(path, 'r', encoding='utf8') as f:
            tokens = 0
            lines = f.readlines()
            for line in lines:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if train:
                    for word in words:
                        self.dictionary.add_word(word)
        with open(path, 'r', encoding='utf8') as f:
            ids = torch.LongTensor(tokens)
            lines = f.readlines()
            token = 0
            for line in lines:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def batchfiy(self,
                 n_neg=3,
                 gram=2,
                 bsz=4096,
                 subsample=True,
                 model='train'):
        print('batchfiy ...')

        if model == 'train':
            data = self.train

        unigramtable = UnigramTable(data, self.n_tokens)

        if subsample:
            prob = np.zeros(self.n_tokens)
            mask = []
            for w in data:
                prob[w.item()] += 1
            prob /= len(data)
            for i in range(len(prob)):
                prob[i] = (np.sqrt(prob[i] / 0.001) + 1) * 0.001 / prob[i]
            for w in data:
                if np.random.rand() < prob[w.item()]:
                    mask.append(True)
                else:
                    mask.append(False)

        u_pos = torch.LongTensor()
        v_pos = torch.LongTensor()
        u_neg = torch.LongTensor()
        v_neg = torch.LongTensor()

        u_pos_batch = torch.zeros(bsz, dtype=torch.long)
        v_pos_batch = torch.zeros(bsz, dtype=torch.long)
        u_neg_batch = torch.zeros([bsz, n_neg], dtype=torch.long)
        v_neg_batch = torch.zeros([bsz, n_neg], dtype=torch.long)

        vs = torch.LongTensor(2 * gram)
        nb = 0
        for i in tqdm(range(gram, len(data) - gram)):

            if nb == bsz:
                u_pos = torch.cat((u_pos, torch.unsqueeze(u_pos_batch, 0)))
                v_pos = torch.cat((v_pos, torch.unsqueeze(v_pos_batch, 0)))
                u_neg = torch.cat((u_neg, torch.unsqueeze(u_neg_batch, 0)))
                v_neg = torch.cat((v_neg, torch.unsqueeze(v_neg_batch, 0)))
                nb = 0

            if subsample and not mask[i]:
                continue
            u = data[i].item()
            vs[:gram] = data[i - gram:i]
            vs[gram:] = data[i + 1:i + gram + 1]
            for v in vs:
                if subsample and not mask[v.item()]:
                    continue
                u_pos_batch[nb] = u
                v_pos_batch[nb] = v.item()
                u_neg_batch[nb] = torch.LongTensor([u] * n_neg)
                v_neg_batch[nb] = torch.LongTensor(unigramtable.sample(n_neg, v))

            nb += 1

        return list(zip(u_pos, v_pos, u_neg, v_neg))


def similar_words(dictionary, embeddings, words, num=5):
    for word in words:
        tag = dictionary.word2idx[word]
        tag_emb = embeddings[tag]
        dot = torch.mv(embeddings, tag_emb)
        tag_norm = torch.norm(tag_emb, 2)
        sim = torch.div(dot, tag_norm)
        sim = torch.div(sim, torch.norm(embeddings, 2, 1))
        dist, idxs = torch.topk(sim, num)
        neighbors = []
        for idx in idxs:
            neighbors.append(dictionary.idx2word[idx])
        print('target word is : {}, similar words are : {}'.format(word, neighbors))


def save_vectors(dictionary, embeddings):
    # with open('results/skip_gram.2M.100d.txt', 'w', encoding='utf8') as f:
    #     for i in range(len(dictionary)):
    #         word = dictionary.idx2word[i]
    #         vec = word
    #         for j in embeddings[i]:
    #             vec += ' {:.6f}'.format(j)
    #         f.write(vec)
    #         f.write('\n')
    res = (dictionary.word2idx, embeddings, embeddings.shape[1])
    torch.save(res, 'results/skip_gram.2M.100d.txt.pt')
