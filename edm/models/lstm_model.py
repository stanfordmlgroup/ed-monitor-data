import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmModel(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(LstmModel, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.hidden2class = nn.Linear(hidden_dim, 1)

    def forward(self, input, seq_lens):
        x = pack_padded_sequence(input, seq_lens, batch_first=True, enforce_sorted=False)
        x, (ht, ct) = self.lstm(x)
        lstm_out, input_sizes = pad_packed_sequence(x, batch_first=True, total_length=input.size(1))
        last_seq_idxs = torch.tensor(seq_lens - 1).to(lstm_out.get_device())
        lstm_out = lstm_out[range(lstm_out.shape[0]), last_seq_idxs, :]
        out = self.hidden2class(lstm_out[:, :])
        return out
