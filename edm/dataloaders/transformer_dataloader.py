from torch.utils.data import Dataset
import numpy as np


class TransformerDataLoader(Dataset):
    def __init__(self, waveform_df):
        self.waveform_df = waveform_df

    def __len__(self):
        return len(self.waveform_df)
    
    def get_labels(self):
        return self.waveform_df["outcome"].to_numpy().astype(np.int64)

    def __getitem__(self, idx):
        patient_id = self.waveform_df.iloc[[idx]]["patient_id"].item()
        cls = self.waveform_df.iloc[[idx]]["outcome"].item()

        cols_to_drop = ["patient_id", "outcome"]
        dims = self.waveform_df.iloc[[idx]].drop(cols_to_drop, axis=1)
        dims = dims.to_numpy()
#         output = np.zeros(2)
#         output[int(cls)] = 1
#         return dims, output, patient_id
        return dims, int(cls), patient_id

    