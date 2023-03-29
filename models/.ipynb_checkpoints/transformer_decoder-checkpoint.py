from torch.nn import TransformerDecoderLayer, TransformerDecoder, LayerNorm
import torch.nn.functional as F
from torch import nn

class decoder(nn.Module):
    def __init__(self, dim_conf, num_classes):
        super().__init__()
        self.embedding = nn.Linear(dim_conf, 1536)
        decoder_layer = TransformerDecoderLayer(d_model=1536, 
                                                nhead=8, 
                                                dim_feedforward=2048, 
                                                dropout=0.1,                                                    
                                                activation=F.relu, 
                                                layer_norm_eps=1e-5, 
                                                batch_first=True,
                                                norm_first=False)
        decoder_norm = LayerNorm(1536, 1e-5)
        self.decoder = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.predictor = nn.Linear(1536, num_classes)
        
    def forward(self, x, memory):
        x = self.embedding(x).unsqueeze(1)
        x = self.decoder(x, memory).squeeze(1)
        x = self.predictor(x)
        return x