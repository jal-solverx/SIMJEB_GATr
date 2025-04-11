from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatr.layers import EquiLinear, EquiLayerNorm, GATrBlock, GeoMLP, SelfAttentionConfig, MLPConfig
from gatr.utils.tensors import construct_reference_multivector
from utils import encode_to_PGA, extract_from_PGA, von_mises


class GATrmk3(nn.Module):
    """
    GATr: A modified version of the Geometric Algebra Transformer

    What's changed? Instead of using one GeoMLP layer with one GeoAttention layer, 
    we will use around 2 GeoMLP layers for every Geo Attention layer
    """
    def __init__(
        self,
        in_mv_channel : int,
        in_s_channel : int,
        n_layers_gatr : int,
        n_mlp_per_gatr : int, 
        n_attn_heads : int,
        hidden_mv_channel: int,
        hidden_s_channel : int,
        out_mv_channel : int,
        out_s_channel : int
    ):
        super().__init__()

        # dimension not specified as we only use multivectors 
        self.n_layers_gatr = n_layers_gatr
        self.n_mlp_per_gatr = n_mlp_per_gatr
        self.in_mv_channel = in_mv_channel
        self.in_s_channel = in_s_channel
        self.hidden_mv_channel = hidden_mv_channel
        self.hidden_s_channel = hidden_s_channel
        self.out_mv_channel = out_mv_channel
        self.out_s_channel = out_s_channel

        '''
        (1) Encoder
          In GATr this is a single Equilinear layer
        '''

        self.encoders = nn.ModuleList([
            EquiLinear(in_mv_channels=self.in_mv_channel, 
                       out_mv_channels=self.hidden_mv_channel, 
                       in_s_channels=self.in_s_channel, 
                       out_s_channels= self.hidden_s_channel)
            for _ in range(1)
        ])
        

        '''
        (2) Processor
          - This is the main GATr block
        '''
        self.norm = EquiLayerNorm()

        self.processor = nn.ModuleList(
            # [GeoMLP(MLPConfig(
            #         mv_channels = (self.hidden_mv_channel, 2 * self.hidden_mv_channel, self.hidden_mv_channel),
            #         s_channels = (self.hidden_s_channel, 2 * self.hidden_s_channel, self.hidden_s_channel),
            #         dropout_prob=0.0
            #     )
            # ) for _ in range(self.n_mlp_per_gatr)]
            # +
            [GATrBlock(
                mv_channels=self.hidden_mv_channel,
                s_channels=self.hidden_s_channel,
                n_mlp = self.n_mlp_per_gatr,
                attention=SelfAttentionConfig(
                    in_mv_channels = self.hidden_mv_channel,
                    out_mv_channels = self.hidden_mv_channel,
                    in_s_channels = self.hidden_s_channel,
                    out_s_channels = self.hidden_s_channel,
                    num_heads=n_attn_heads,
                    dropout_prob=0.0,
                ),
                mlp=MLPConfig(
                    mv_channels = self.hidden_mv_channel,
                    s_channels = self.hidden_s_channel,
                    dropout_prob=0.0,
                ),
                dropout_prob=0.0
            )
            for _ in range(self.n_layers_gatr)]
        )


        '''
        (3) Decoder
          In GATr this is a single Equilinear layer
        '''

        self.decoders = nn.ModuleList([
            EquiLinear(in_mv_channels=self.hidden_mv_channel, 
                       out_mv_channels=self.out_mv_channel, 
                       in_s_channels = self.hidden_s_channel, 
                       out_s_channels=self.out_s_channel)
            for _ in range(1)
        ])

    def forward(self, graph: Data):

        # (1) Encoder

        x_PGA, bc = encode_to_PGA(graph)


        # (2) Processor
        #x_PGA = x_PGA.squeeze(0).squeeze(1)
        reference_mv = construct_reference_multivector('data', x_PGA) # reference MV needed for the equivariant join layer
        for encoder in self.encoders:
            x_PGA, bc = encoder(x_PGA, bc)
        for index, processor in enumerate(self.processor):
            # if index < self.n_mlp_per_gatr:
            #     x_PGA, bc = self.norm(x_PGA, bc)
            #     x_PGA, bc = processor(x_PGA, scalars = bc, reference_mv = reference_mv)
            #     #print(x_PGA, bc)
            # else:
            x_PGA, bc = processor(x_PGA, scalars = bc, reference_mv = reference_mv)
                #print(x_PGA, bc)
        for decoder in self.decoders:
            x_PGA, bc = decoder(x_PGA, bc)

        # (3) Decoder
        y = extract_from_PGA(x_PGA, bc, graph)
        return y.bc
        von_mises_stress = von_mises(y.x[:, 6:9])

        return von_mises_stress