import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentAnalysis(nn.Module):
    def __init__(self, args):
        super(SentimentAnalysis, self).__init__()
        self.embedding = nn.Embedding(args.input_dim, args.embed_dim)
        self.rnn = nn.LSTM(
            args.embed_dim,
            args.hidden_dim,
            num_layers=args.n_layers,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
            )
        self.fc = nn.Linear(args.hidden_dim*2, args.output_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden.squeeze(0))


class FastText(nn.Module):
    def __init__(self, args):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(args.input_dim, args.embed_dim)
        self.fc = nn.Linear(args.embed_dim, args.output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.max_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(args.input_dim, args.embed_dim)
        # self.conv0 = nn.Conv2d(in_channels=1, out_channels=args.n_filters, kernel_size=(args.filter_sizes[0], args.embed_dim))
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=args.n_filters, kernel_size=(args.filter_sizes[1], args.embed_dim))
        # self.conv2 = nn.Conv2d(in_channels=1, out_channels=args.n_filters, kernel_size=(args.filter_sizes[2], args.embed_dim))
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=args.n_filters, kernel_size=(i, args.embed_dim)) for i in args.filter_sizes])
        self.fc = nn.Linear(len(args.filter_sizes)*args.n_filters, args.output_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)

        # conved0 = F.relu(self.conv0(embedded).squeeze(3))
        # conved1 = F.relu(self.conv1(embedded).squeeze(3))
        # conved2 = F.relu(self.conv2(embedded).squeeze(3))
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # pooled0 = F.max_pool1d(conved0, conved0.shape[2]).squeeze(2)
        # pooled1 = F.max_pool1d(conved1, conved1.shape[2]).squeeze(2)
        # pooled2 = F.max_pool1d(conved2, conved2.shape[2]).squeeze(2)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)

