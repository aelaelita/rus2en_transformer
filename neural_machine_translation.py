import torch
from torch import nn


class NMTEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NMTEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, x, hidden):
        out = self.embedding(x)
        out = out.permute((1, 0, 2))
        out, hidden = self.gru(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_dim))


class NMTDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_length=100):
        super(NMTDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.attention = NMTAttention(hidden_dim, max_length)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden, encoder_output):
        embedded = self.embedding(x)
        encoder_output = encoder_output
        out = self.attention(embedded[0], hidden[0], encoder_output)
        out, hidden = self.gru(out, hidden)
        out = self.linear(out)
        out = self.softmax(out[0])
        return out, hidden


class NMTAttention(nn.Module):
    def __init__(self, hidden_dim, max_length):
        super(NMTAttention, self).__init__()
        self.linear1 = nn.Linear(hidden_dim * 2, max_length)
        self.softmax = nn.Softmax(dim=1)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, embedded, hidden, encoder_output):
        out = torch.cat((embedded, hidden), 1)
        out = self.linear1(out)
        out = self.softmax(out).unsqueeze(0)
        out = torch.bmm(out, encoder_output)[0]
        out = torch.cat((embedded, out), 1)
        out = self.linear2(out).unsqueeze(0)
        out = self.relu(out)
        return out
