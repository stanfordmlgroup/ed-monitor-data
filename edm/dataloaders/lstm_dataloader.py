from torch.utils.data import Dataset
import numpy as np


class LstmDataLoader(Dataset):
    def __init__(self, mat, ids, seq_lens, y):
        self.mat = mat
        self.ids = ids
        self.seq_lens = seq_lens
        self.y = y

    def __len__(self):
        return len(self.mat)
    
    def get_labels(self):
        return np.array(self.y).astype(np.int64)

    def __getitem__(self, idx):
        patient_id = self.ids[idx]
        seq_len = self.seq_lens[idx]
        cls = self.y[idx]

        dims = self.mat[idx]
        return dims, seq_len, int(cls), patient_id
