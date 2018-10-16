import argparse
import time
import pickle
import profile
import torch
import torch.optim as optim

from utils import TextData, similar_words, save_vectors
from model import SkipGramModel


def train(model, optimizer, train_data, epoch):
    n_batch = len(train_data)
    model.train()
    start_time = time.time()
    for idx, (u_pos, v_pos, u_neg, v_neg) in enumerate(train_data):
        model.zero_grad()

        loss = model(u_pos, v_pos, u_neg, v_neg)
        # loss = model(batch[0], batch[1], batch[2])
        loss.backward()
        optimizer.step()

        if idx % 50 == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | loss {:5.2f} '
                .format(epoch, idx, n_batch, elapsed * 20, loss.item()))
            start_time = time.time()


def evaluate(model, eval_data):
    pass


def save(model):
    torch.save(model.state_dict(), 'params.pkl')
    print('saved parameters ...')
    # model.load_state_dict(torch.load('params.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--path', type=str, default='./.data/wikitext-2/')
    parser.add_argument('--n_hidden', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--gram', type=int, default=2)
    parser.add_argument('--tie', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--subsample', type=bool, default=True)
    parser.add_argument('--n_negative', type=int, default=3)
    args = parser.parse_args()

    data = TextData(args.path)
    # # profile.run('.data.batchfiy(n_neg=args.n_negative, gram=args.gram, bsz=args.batch_size, subsample=args.subsample)')
    # train_data = data.batchfiy(n_neg=args.n_negative, gram=args.gram, bsz=args.batch_size, subsample=args.subsample)
    #
    # # with open('batches.pkl', 'wb') as f: ??????
    # # with open('.data.pkl', 'rb') as f:
    # #     .data = pickle.load(f)
    #
    # # eval_data = .data.evaluate
    #
    # model = SkipGramModel(data.n_tokens, args.n_hidden)
    # # model.load_state_dict(torch.load('params.pkl'))
    #
    # optimizer = optim.SparseAdam(model.parameters())
    #
    # for epoch in range(args.epochs):
    #     train(model, optimizer, train_data, epoch)
    #     # evaluate(model, eval_data)
    # save(model)

    params = torch.load('results/params.pkl')
    similar_words(data.dictionary, params['u_embeddings.weight'], ['China', 'one'])
    save_vectors(data.dictionary, params['u_embeddings.weight'])
