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
    def __init__(self, hidden_dim, output_dim):
        super(NMTDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.attention = NMTAttention(hidden_dim)
        self.gru = nn.GRU(2 * hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, encoder_output):
        encoder_output = encoder_output.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)
        embedded = self.embedding(x)
        out = self.attention(embedded, hidden, encoder_output)
        out, hidden = self.gru(out)
        out = out.squeeze(1)
        out = self.linear(out)
        return out, hidden


class NMTAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(NMTAttention, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, embedded, hidden, encoder_output):
        out = self.relu(self.linear1(encoder_output) + self.linear2(hidden))
        out = self.linear3(out)
        out = self.softmax(out)
        context = torch.sum(out * encoder_output, dim=1).unsqueeze(1)
        out = torch.cat((context, embedded), -1)
        return out
