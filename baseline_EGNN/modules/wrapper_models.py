from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layer import GATv2, GCN, MLP, EGNN

class baselineEGNN(nn.Module):
    def __init__(
        self,
        input_dim : int,
        output_dim : int,
        hidden_dim : int,
        n_layers_proc : int
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers_proc = n_layers_proc

        self.mid_dim = self.hidden_dim // 2
        self.quad_dim = self.hidden_dim // 4
        self.oct_dim = self.hidden_dim // 8

        '''
        (1) Encoder
            EGNN으로 입력 차원에서 숨겨진 차원으로 팽창하기기
        '''

        self.encoder = EGNN(
            input_nf = self.input_dim,
            hidden_nf = self.mid_dim,
            output_nf = self.hidden_dim
        )

        # self.encoder_ln = nn.LayerNorm(self.hidden_dim)

        '''
        (2) Processor
          - default ver: EGNN
        '''

        self.processor = nn.ModuleList(
            [EGNN(
                input_nf = self.hidden_dim,
                hidden_nf = self.mid_dim,
                output_nf = self.hidden_dim
            ) for _ in range(self.n_layers_proc)]
        )

        # self.processor = GATv2(
        #         input_dim = self.hidden_dim,
        #         hidden_dim = self.hidden_dim,
        #         output_dim = self.hidden_dim,
        #         num_gnn_layers = self.n_layers_pro,
        #         residual = True,
        #         dropout = 0.,
        #         edge_dim = 4,
        #     )

        '''
        (3) Processor
          - default ver: EGNN
        '''

        self.decoder = EGNN(
                input_nf = self.hidden_dim,
                hidden_nf = self.mid_dim,
                output_nf = self.output_dim
        )

    def forward(self, graph: Data):

        # (0) data extract
        x = graph.x
        pos = graph.x[:, :3] # 좌표값
        bc = graph.bc

        edge_index = graph.edge_index
        edge_weight = graph.edge_weight

        # (1) Encoder

        hidden_h, hidden_x = self.encoder(x = bc, coord = pos, edge_index = edge_index, edge_attr = None)

        # (2) Processor
        for layer in self.processor:
            hidden_h, hidden_x = layer(x=hidden_h, coord = hidden_x, edge_index=edge_index, edge_attr = None)

        # (3) Decoder
        result, _ = self.decoder(x=hidden_h, coord=hidden_x, edge_index=edge_index, edge_attr = None)

        return result