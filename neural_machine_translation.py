import torch
from torch import nn


class NMTEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NMTEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)

    def forward(self, x, hidden):
        out = self.embedding(x).view(1, 1, -1)
        out, hidden = self.lstm(out)
        return out, hidden


class NMTDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_length=100):
        super(NMTDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.attention = NMTAttention(hidden_dim, max_length)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden, encoder_output):
        embedded = self.embedding(x).view(1, 1, -1)
        encoder_output = encoder_output.unsqueeze(0)
        out = self.attention(embedded[0], hidden[0], encoder_output)
        out, hidden = self.lstm(out, hidden)
        out = self.linear(out)
        out = self.softmax(out[0])
        return out, hidden


class NMTAttention(nn.Module):
    def __init__(self, hidden_dim, max_length):
        super(NMTAttention, self).__init__()
        self.linear1 = nn.Linear(hidden_dim * 2, max_length)
        self.softmax = nn.Softmax()
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, embedded, hidden, encoder_output):
        out = torch.cat((embedded, hidden), 1)
        out = self.linear1(out)
        out = self.softmax(out, dim=1).unsqueeze(0)
        out = torch.bmm(out, encoder_output)[0]
        out = torch.cat((embedded, out), 1)
        out = self.linear2(out).unsqueeze(0)
        out = self.relu(out)
        return out
