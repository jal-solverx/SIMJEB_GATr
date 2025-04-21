import torch
import torch.nn as nn
from typing import Optional
from torch.nn import Identity
from torch_geometric.nn import GCNConv, GATv2Conv, LayerNorm, InstanceNorm, GraphNorm, BatchNorm, MessagePassing
import torch.nn.functional as F

def get_gnn_norm(norm_type):
    if norm_type in [ 'layer', 'layernorm' ,  'LayerNorm']:
        return LayerNorm
    elif norm_type in [ 'instance', 'instancenorm' ,  'InstanceNorm']:
        return InstanceNorm
    
    elif norm_type in [ 'batch', 'batchnorm' ,  'BatchNorm']:
        return BatchNorm
    
    elif norm_type in ['graph', 'graphnorm', 'GraphNorm']:
        return GraphNorm
    else :
        return Identity

class EGNN(MessagePassing):
    # MessagePassing은 PyTorch Geometric에서 제공하는 클래스.
    # 그래프 상에서 "메시지 패싱" 연산을 쉽게 구현할 수 있게 해주는 부모 클래스.
    # 여기서는 'add' 방식으로 메시지를 합산(aggregation)할 것임.
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, 
                 act_fn=nn.SiLU(), residual=True, attention=False, 
                 normalize=False, coords_agg='mean', tanh=False):
        super().__init__(aggr='add')  # 'add'는 메시지를 합하는 방식
        # input_nf : 입력 노드 특성 차원 수
        # output_nf: 출력 노드 특성 차원 수
        # hidden_nf: 내부적으로 메시지나 노드 특성을 변환할 때 사용하는 은닉 차원 수
        # edges_in_d: 엣지 특성 차원 수 (없으면 0)
        # act_fn: 활성함수 (SiLU: swish같은 활성화)
        # residual: 잔차연결 사용 여부
        # attention: 메시지에 어텐션 기법 적용 여부
        # normalize: 좌표차이를 정규화할지 여부
        # coords_agg: 좌표 업데이트 시 어떤 방식으로 엣지 정보를 모을지('mean'이면 평균)
        # tanh: 마지막에 tanh 활성화 사용 여부

        input_edge = input_nf * 2  # 엣지 메시지 입력은 (x_i, x_j)를 이어붙이므로 x 차원은 2배
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1  # 거리 제곱(radial) 1차원 추가

        # 입력과 출력 노드 특성이 다르면 residual 연결용으로 차원 맞추는 MLP 만듦
        if input_nf != output_nf:
            self.residual_mlp = nn.Linear(input_nf, output_nf, bias=False)
            self.act = act_fn
        else:
            self.residual_mlp = None

        # edge_mlp: 엣지 메시지를 만드는 MLP
        # 입력: [x_i, x_j, radial(거리), (edge_attr)]
        # 차원: (input_nf*2 + 1 + edges_in_d) -> hidden_nf -> hidden_nf
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # node_mlp: 노드 특성 업데이트용 MLP
        # 입력: [현재 노드 특성, 엣지 메시지 합(out_message)]
        # 차원: (hidden_nf + input_nf) -> hidden_nf -> output_nf
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        # 좌표 업데이트를 위한 계층
        # layer는 hidden_nf -> 1 로 가는 선형계층
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        # coord_mlp: 좌표 업데이트 계층
        # 입력: 엣지 메시지(hidden_nf차원)
        # 출력: 스칼라값(1차원), 이 스칼라로 coord_diff를 곱해 좌표업데이트 벡터 만들기
        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer
        ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # 어텐션을 쓸 경우, m_ij에 대해서 어텐션 가중치 계산할 MLP
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def coord2radial(self, edge_index, coord):
        # edge_index: [2, E] 형태 (E는 엣지 수)
        # coord: [N, 3] 형태 (N은 노드 수)
        # 이 함수는 엣지마다 두 노드의 좌표 차이와 거리 제곱을 구함.
        row, col = edge_index
        coord_diff = coord[row] - coord[col]  # [E, 3]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)  # [E,1] 거리 제곱

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm  # 거리로 나눠 단위벡터화

        return radial, coord_diff

    def forward(self, x, edge_index, coord, edge_attr, face=None):
        # x: [N, input_nf]
        # coord: [N, 3]
        # edge_index: [2, E]
        # edge_attr: [E, edges_in_d] or None

        # 1) 각 엣지별 radial, coord_diff 계산
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # 2) propagate 호출 -> 내부적으로 message 함수가 각 엣지마다 호출되고,
        #    aggregate로 각 노드별로 합산
        # 반환: out = [N, 3 + output_nf] 형태로 생각할 수 있음 (추후 분리)
        # 실제로는 aggregate 단계에서 (c_ij, m_ij)를 따로 합산한 뒤 합쳐서 반환
        out = self.propagate(edge_index, x=x, radial=radial, edge_attr=edge_attr, coord_diff=coord_diff)

        # out은 cat([coord_update, message_update], dim=1) 형태.
        # 여기서 coord_update: [N, 3], message_update: [N, hidden_nf or output_nf]
        out_coord, out_message = out[:, :3], out[:, 3:]

        # 3) 좌표 업데이트
        # 기존 coord에 out_coord를 더한다.
        coord = coord + out_coord

        # 4) 노드 특성 업데이트
        # out_message와 기존 x를 합쳐 node_mlp에 통과
        new_x = self.node_mlp(torch.cat([x, out_message], dim=1))
        # 만약 residual이면 기존 x와 new_x를 합쳐 최종 out을 만든다.
        if self.residual:
            if self.residual_mlp is not None:
                # 차원이 안맞을 경우 residual_mlp로 변환 후 더함
                out = self.act(self.residual_mlp(x) + new_x)
            else:
                # 동일 차원이면 단순 더함
                out = x + new_x
        else:
            out = new_x

        # 최종 반환: 업데이트된 노드 특성 out, 업데이트된 좌표 coord
        return out, coord

    def message(self, x_i, x_j, radial, edge_attr, coord_diff):
        # message 함수는 각 엣지별로 호출됨
        # x_i, x_j: 엣지의 두 끝 노드 특성
        # radial: 엣지에 대한 거리 제곱 값 [E,1]
        # edge_attr: 엣지 특성 [E, edges_in_d]
        # coord_diff: 좌표차이 [E,3]

        # m_ij 만들기: [x_i, x_j, radial, edge_attr]를 연결해 edge_mlp 통과
        if edge_attr is None:
            m_ij = torch.cat([x_i, x_j, radial], dim=1)
        else:
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            m_ij = torch.cat([x_i, x_j, radial, edge_attr], dim=1)

        # edge_mlp로 메시지 변환: (input_nf*2 + 1 + edges_in_d) -> hidden_nf -> hidden_nf
        m_ij = self.edge_mlp(m_ij)

        # 어텐션 사용 시, m_ij를 통해 가중치(att_val) 계산 후 곱해줌
        if self.attention:
            att_val = self.att_mlp(m_ij)  # [E,1]
            m_ij = m_ij * att_val

        # 좌표 업데이트용 스칼라 계산: coord_mlp(m_ij) -> [E,1]
        # 이를 coord_diff에 곱해 c_ij 만듦
        # c_ij: 각 엣지가 노드 좌표 변경에 기여하는 벡터 [E,3]
        c_ij = coord_diff * self.coord_mlp(m_ij)

        # 반환: (c_ij, m_ij)
        # 나중에 aggregate 단계에서 이걸 노드 단위로 합산함
        return torch.cat([c_ij, m_ij], dim=-1)

    def aggregate(self, inputs, index,
                  ptr= None,
                  dim_size = None):
        # aggregate는 부모클래스(MessagePassing)의 메서드 재정의
        # inputs는 message 함수 결과 [E, (3+hidden_nf)] 형태
        # index는 각 엣지가 연결하는 노드 인덱스
        # 이 단계에서 같은 노드로 들어오는 모든 (c_ij, m_ij)를 합산

        c = self.aggr_module(inputs[:, :3], index, ptr=ptr, dim_size=dim_size, dim=0)
        # 여기서 c는 좌표 업데이트용 벡터들의 합 [N, 3]

        m = self.aggr_module(inputs[:, 3:], index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)
        # m은 메시지 벡터들의 합 [N, hidden_nf]

        # 최종 (c,m)을 concat해 반환: [N, 3 + hidden_nf]
        return torch.cat([c, m], dim=1)

class GATv2(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_gnn_layers: int,
                 residual: bool = False,
                 norm_type: str = 'layer',
                 activation_fn: nn.Module = nn.GELU(),
                 dropout: float = 0.0,
                 edge_dim: int = None,
                 n_heads: int = 4):
        super(GATv2, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.hidden_dim = hidden_dim

        self.min_dim = int(hidden_dim / n_heads)
        norm_layer = get_gnn_norm(norm_type)
        
        self.conv_layers = nn.ModuleList()
        self.gnn_norm_layers = nn.ModuleList()
        
        if num_gnn_layers == 1:
            self.conv_layers.append(
                GATv2Conv(input_dim, int(output_dim / n_heads), heads=n_heads, edge_dim=edge_dim, concat=True, dropout=dropout)
                )
            self.gnn_norm_layers.append(norm_layer(output_dim))
        else:
            self.conv_layers.append(
                GATv2Conv(input_dim, self.min_dim, heads=n_heads, edge_dim=edge_dim, concat=True, dropout=dropout)
                )
            self.gnn_norm_layers.append(norm_layer(hidden_dim))
            for _ in range(num_gnn_layers - 2):
                self.conv_layers.append(
                    GATv2Conv(hidden_dim, self.min_dim, heads=n_heads, edge_dim=edge_dim, concat=True, dropout=dropout)
                    )
                self.gnn_norm_layers.append(norm_layer(hidden_dim))
            self.conv_layers.append(
                GATv2Conv(hidden_dim, int(output_dim / n_heads), heads=n_heads, edge_dim=edge_dim, concat=True, dropout=dropout)
                )
            self.gnn_norm_layers.append(norm_layer(output_dim))
            
        self.activation_fn = activation_fn
        self.residual = residual

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        for i in range(self.num_gnn_layers):
            x_res = x
            x = self.conv_layers[i](x, edge_index, edge_attr)
            if self.residual:
                x = x + x_res
            x = self.gnn_norm_layers[i](x)
            x = self.activation_fn(x)
        return x


class GCN(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_gnn_layers: int,
                 residual: bool = False,
                 norm_type: str = 'layer',
                 activation_fn: nn.Module = nn.GELU(),
                 dropout: float = 0.0):
        super(GCN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        norm_layer = get_gnn_norm(norm_type)
        
        self.conv_layers = nn.ModuleList()
        self.gnn_norm_layers = nn.ModuleList()
        
        if num_gnn_layers == 1:
            self.conv_layers.append(
                GCNConv(in_channels=input_dim, out_channels=output_dim, add_self_loops=True)
            )
            self.gnn_norm_layers.append(norm_layer(output_dim))
        else:
            self.conv_layers.append(
                GCNConv(in_channels=input_dim, out_channels=hidden_dim, add_self_loops=True)
            )
            self.gnn_norm_layers.append(norm_layer(hidden_dim))
            for _ in range(num_gnn_layers - 2):
                self.conv_layers.append(
                    GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, add_self_loops=True)
                )
                self.gnn_norm_layers.append(norm_layer(hidden_dim))
            self.conv_layers.append(
                GCNConv(in_channels=hidden_dim, out_channels=output_dim, add_self_loops=True)
            )
            self.gnn_norm_layers.append(norm_layer(output_dim))
            
        self.activation_fn = activation_fn
        self.residual = residual

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        for i in range(self.num_gnn_layers):
            x_res = x
            x = self.conv_layers[i](x, edge_index, edge_weight)
            if self.residual:
                x = x + x_res
            x = self.gnn_norm_layers[i](x)
            x = self.activation_fn(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float = 0.0):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.lins = nn.ModuleList()
        
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x