import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Transformer parameters
# Copied from the PRNA model
#
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers
dropout_rate = 0.2
model_name = 'ctn'
classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006',
                  '713426002', '445118002', '39732003', '164909002', '251146004',
                  '698252002', '10370003', '284470004', '427172004', '164947007',
                  '111975006', '164917005', '47665007', '427393009',
                  '426177001', '426783006', '427084000', '164934002',
                  '59931005'])


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers
    '''

    def __init__(self, d_model, h, d_ff, num_layers, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1)

        encode_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.h,
            dim_feedforward=self.d_ff,
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.pe(out)
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = out.mean(0)  # global pooling
        return out


# 15 second model
class CTN(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_patient_feats, classes, leads=1):
        super(CTN, self).__init__()

        self.encoder = nn.Sequential(  # downsampling factor = 20
            nn.Conv1d(leads, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.transformer = Transformer(d_model, nhead, d_ff, num_layers, dropout=0.1)
        self.fc1 = nn.Linear(d_model, deepfeat_sz)
        self.fc2 = nn.Linear(deepfeat_sz + nb_patient_feats, len(classes))
        self.dropout = nn.Dropout(dropout_rate)
        self.nb_patient_feats = nb_patient_feats
        
        print(f"deepfeat_sz={deepfeat_sz}, nb_patient_feats={nb_patient_feats}")

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.apply(_weights_init)

    def forward(self, x, wide_feats):
        z = self.encoder(x)  # encoded sequence is batch_sz x nb_ch x seq_len
        out = self.transformer(z)  # transformer output is batch_sz x d_model

        # out = self.dropout(F.relu(self.fc1(out)))  
        out = F.relu(self.fc1(out))
        if self.nb_patient_feats > 0:
            wide_feats = torch.squeeze(wide_feats).float()
            out = self.fc2(torch.cat([wide_feats, out], dim=1))
        else:
            out = self.fc2(out)

        return out


def load_best_model(model_loc, deepfeat_sz, remove_last_layer=True, leads=1, output_classes=classes, nb_patient_feats=0):
    model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_patient_feats, output_classes, leads).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if model_loc is not None:
        if torch.cuda.is_available():
            checkpoint = torch.load(model_loc)
        else:
            checkpoint = torch.load(model_loc, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loading best model: best_loss', checkpoint['best_loss'], 'best_auroc', checkpoint['best_auroc'],
              'at epoch',
              checkpoint['epoch'])
    else:
        print("NOT using a pre-trained model")

        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if remove_last_layer:
        model = torch.nn.Sequential(*(list(list(model.children())[0].children())[:-2]))
    return model
