import torch
import torch.nn as nn
from config import Q_INPUT_DIM, SEQUENCE_LENGTH, Q_EMBEDDING_DIM, NHEAD, Q_NUM_LAYERS, Q_OUTPUT_DIM, Q_DROPOUT

class TransformerQNet(nn.Module):
    def __init__(self, input_dim=Q_INPUT_DIM, seq_len=SEQUENCE_LENGTH, d_model=Q_EMBEDDING_DIM, nhead=NHEAD, num_layers=Q_NUM_LAYERS, output_dim=Q_OUTPUT_DIM, dropout=Q_DROPOUT):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(seq_len * d_model, output_dim)
    def forward(self, x):
        x = self.input_linear(x) + self.pos_encoding
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        return self.output_layer(x)
