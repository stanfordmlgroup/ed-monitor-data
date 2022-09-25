from torch.utils.data import Dataset
import numpy as np


class TransformerDataLoader(Dataset):
    def __init__(self, waveform_df, num_numerics_cols):
        self.waveform_df = waveform_df
        self.num_numerics_cols = num_numerics_cols

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
        
        # Take the last columns to be the numerics columns
        if self.num_numerics_cols > 0:
            numerics_cols = dims[:, -self.num_numerics_cols:]
            dims = dims[:, :-self.num_numerics_cols]
            return numerics_cols, dims, int(cls), patient_id
        else:
            return np.array([]), dims, int(cls), patient_id

    