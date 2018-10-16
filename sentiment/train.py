import random
import time
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

from model import SentimentAnalysis, FastText, CNN
from utils import makedirs, get_args

SEED = 1234
torch.manual_seed(SEED)
train_log_template = ' '.join('Time:{:>26s}, Epoch:{:>3.0f}, Batch:{:>5.0f}/{:<5.0f}, Loss:{:>3.5f}, Acc:{:<2.2f}%'.split(','))
log_template = ' '.join('Evaluate - epoch:{:>3.0f}, Train Loss:{:>3.5f} ,Train Acc:{:>2.2f}%, Val Loss:{:>3.5f} ,Val Acc:{:>2.2f}%'.split(','))


# may be no effect
def generate_grams(x):
    n = 2
    # n_grams = set(zip(*[x[i:] for i in range(n)]))
    n_grams = set(zip(*[x[i:] for i in range(n)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def load_data(args):
    text = data.Field(tokenize='spacy', lower=True)
    label = data.LabelField(tensor_type=torch.FloatTensor)
    train, test = datasets.IMDB.splits(text, label)
    # train, valid = train.split(random_state=random.seed(SEED), split_ratio=0.8)
    text.build_vocab(train, max_size=args.vocab_size)
    if args.word_vectors:
        if args.word_vectors == 'skip_gram.2M.100d':
            stoi, vectors, dim = torch.load('.vector_cache/skip_gram.2M.100d.txt.pt')
            text.vocab.set_vectors(stoi, vectors, dim)
        else:
            if os.path.isfile(args.vector_cache):
                text.vocab.vectors = torch.load(args.vector_cache)[:args.input_dim, :]
            else:
                text.vocab.load_vectors(args.word_vectors)
                makedirs(os.path.dirname(args.vector_cache))
                torch.save(text.vocab.vectors, args.vector_cache)

    label.build_vocab(train)

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train, test),
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False,
    )
    pretrained_embeddings = text.vocab.vectors
    assert pretrained_embeddings.shape == torch.Size([args.input_dim, args.embed_dim])
    test_iterator = True
    return train_iterator, valid_iterator, test_iterator, pretrained_embeddings


def train(model, train_iterator, optimizer, criterion, **kwargs):
    train_loss = 0.
    train_acc = 0.
    model.train()
    for batch_idx, batch in enumerate(train_iterator):
        optimizer.zero_grad()
        output = model(batch.text).squeeze(1)
        loss = criterion(output, batch.label)

        train_loss += loss.item()
        acc = (torch.round(nn.Sigmoid()(output)) == batch.label).sum().item() / len(output) * 100
        train_acc += acc

        loss.backward()
        optimizer.step()

        print(train_log_template.format(
            time.asctime(),
            kwargs['epoch']+1,
            batch_idx+1,
            len(train_iterator),
            loss.item(),
            acc,
        ))
    return train_loss / len(train_iterator), train_acc / len(train_iterator)


def evaluate(model, valid_iterator, criterion, **kwargs):
    val_loss = 0.
    val_acc = 0.
    model.eval()
    with torch.no_grad():
        for batch in valid_iterator:
            output = model(batch.text).squeeze(1)
            val_loss += criterion(output, batch.label).item()
            val_acc += (torch.round(nn.Sigmoid()(output)) == batch.label).sum().item() / len(output) * 100
    return val_loss / len(valid_iterator), val_acc / len(valid_iterator)


def save_model(model, **kwargs):
    snapshot_prefix = os.path.join(kwargs['save_path'], 'snapshot')
    snapshot_path = snapshot_prefix + \
        '_acc_{:2.3f}_loss_{:.6f}_epoch_{}_model.pt'.format(kwargs['acc'], kwargs['loss'], kwargs['epoch'])
    torch.save(model, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)


def main():
    best_acc = 0
    args = get_args()
    args.word_vectors = 'skip_gram.2M.100d'
    args.input_dim = args.vocab_size + 2
    makedirs(args.save_path)
    print('loading data...')
    train_iterator, valid_iterator, _, pretrained_embeddings = load_data(args)
    # model = SentimentAnalysis(args)
    # model = FastText(args)
    model = CNN(args)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, epoch=epoch)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, epoch=epoch)

        print(log_template.format(epoch+1, train_loss, train_acc, valid_loss, valid_acc))
        # print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%,'
        #       f' Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        if valid_acc > best_acc:
            save_model(model, save_path=args.save_path, acc=valid_acc, loss=valid_loss, epoch=epoch)


if __name__ == '__main__':
    main()
