import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, n_tokens, n_hidden=100, tie=False):
        super(SkipGramModel, self).__init__()
        self.u_embeddings = nn.Embedding(n_tokens, n_hidden, sparse=True)
        self.v_embeddings = nn.Embedding(n_tokens, n_hidden, sparse=True)
        self.tie = tie
        if self.tie:
            self.u_embeddings.weight = self.v_embeddings.weight

    def init_weights(self):
        initrange = 0.1  # 0.5 / n_embedding
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, u_pos, v_pos, u_neg, v_neg):
        """Forward propagation.

        :param u_pos: center word LongTensor [batch size].
        :param v_pos: neibor words LongTensor [batch size].
        :param u_neg: negatice samples' center word LongTensor [batch size, n_negative].
        :param v_neg: negatice samples LongTensor [batch size, n_negative].
        :return: loss float
        """
        u_emb = self.u_embeddings(u_pos)  # bsz , n_hidden
        v_emb = self.v_embeddings(v_pos)  # bsz , n_hidden
        u_neg_emb = self.u_embeddings(torch.squeeze(u_neg.view(1, -1)))  # bsz * n_neg , n_hidden
        v_neg_emb = self.v_embeddings(torch.squeeze(v_neg.view(1, -1)))  # bsz * n_neg , n_hidden
        loss = torch.mul(u_emb, v_emb)
        loss = torch.sum(loss, 1)
        loss = F.logsigmoid(loss)
        neg_loss = torch.mul(u_neg_emb, v_neg_emb)
        neg_loss = torch.sum(neg_loss, 1)
        neg_loss = F.logsigmoid(-neg_loss)
        return -1 * (torch.sum(loss) + torch.sum(neg_loss))
