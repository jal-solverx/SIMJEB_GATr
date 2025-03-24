from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatr.layers import EquiLinear, GATrBlock, SelfAttentionConfig, MLPConfig
from gatr.utils.tensors import construct_reference_multivector
from utils import encode_to_PGA, extract_from_PGA


class vanillaGATr(nn.Module):
    """
    vanillaGATr: vanilla Graph Attention Transformer for PDEs
    """
    def __init__(
        self,
        n_attn_heads : int,
        n_layers_enc : int,
        n_layers_attn : int,
        hidden_mv_channel: int,
        n_layers_dec : int,
    ):
        super().__init__()

        # dimension not specified as we only use multivectors 
        self.n_layers_enc = n_layers_enc
        self.n_layers_attn = n_layers_attn
        self.n_layers_dec = n_layers_dec
        self.n_hidden_mv_channel = hidden_mv_channel

        '''
        (1) Encoder
          In GATr this is a single Equilinear layer
        '''

        self.encoders = nn.ModuleList([
            EquiLinear(in_mv_channels=1, out_mv_channels=self.n_hidden_mv_channel, in_s_channels= 2, out_s_channels= 2)
            for _ in range(n_layers_enc)
        ])
        

        '''
        (2) Processor
          - This is the main GATr block
        '''

        self.processor = nn.ModuleList([
            GATrBlock(
                mv_channels=self.n_hidden_mv_channel,
                s_channels=2,
                attention=SelfAttentionConfig(
                    in_mv_channels=self.n_hidden_mv_channel,
                    out_mv_channels=self.n_hidden_mv_channel,
                    in_s_channels = 2,
                    out_s_channels = 2,
                    num_heads=n_attn_heads,
                    dropout_prob=0.0,
                ),
                mlp=MLPConfig(
                    mv_channels = self.n_hidden_mv_channel,
                    s_channels = 2,
                    dropout_prob=0.0,
                ),
                dropout_prob=0.0
            )
            for _ in range(n_layers_attn)
        ])


        '''
        (3) Decoder
          In GATr this is a single Equilinear layer
        '''

        self.decoders = nn.ModuleList([
            EquiLinear(in_mv_channels=self.n_hidden_mv_channel, out_mv_channels=1, in_s_channels = 2, out_s_channels=2)
            for _ in range(n_layers_dec)
        ])

    def forward(self, graph: Data):

        # (1) Encoder

        x_PGA, bc = encode_to_PGA(graph)
        #print("type of x_PGA", type(x_PGA))
        #print("type of bc:", type(bc))


        # (2) Processor
        #x_PGA = x_PGA.squeeze(0).squeeze(1)
        reference_mv = construct_reference_multivector('data', x_PGA)
        for encoder in self.encoders:
            x_PGA, bc = encoder(x_PGA, bc)
        for processor in self.processor:
            x_PGA, bc = processor(x_PGA, scalars = bc, reference_mv = reference_mv)
        for decoder in self.decoders:
            x_PGA, bc = decoder(x_PGA, bc)

        # (3) Decoder
        y = extract_from_PGA(x_PGA, bc, graph)

        return y