import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader

def one_hot_from_rbe(rbe2, N_elementf):
    """
    주어진 rbe2 리스트에 있는 element ID들을 이용하여,
    크기가 N_elementf x 1인 one-hot 벡터를 생성하는 함수
    
    Parameters:
        rbe2 (list or numpy array): element ID들이 담긴 리스트
        N_elementf (int): 전체 element의 개수, 즉 one-hot 벡터의 크기
    
    Returns:
        torch.Tensor: N_elementf x 1 크기의 one-hot 벡터
                      해당 element ID에 해당하는 위치에 1이 할당됩니다
    """
    
    # N_elementf x 1 크기의 0으로 채워진 텐서를 생성
    one_hot_vectors = torch.zeros(N_elementf, 1)
    
    # rbe2 리스트의 각 element_id에 대해 해당 인덱스 위치에 1을 할당
    for element_id in rbe2:
        if element_id < N_elementf:  # element_id가 유효한 범위 내에 있는지 확인
            one_hot_vectors[element_id, :] = 1
        else:
            raise ValueError(f"Element ID {element_id} is out of range (N_elementf={N_elementf})")
    
    return one_hot_vectors


def calculate_edge_features(x, edge_index):
    """
    노드 특징 텐서와 edge index를 기반으로, 각 엣지에 대한 특징 벡터를 계산하는 함수
    
    Parameters:
        x (torch.Tensor): 노드 특징 텐서 [num_nodes, 3]
        edge_index (torch.Tensor): 엣지 인덱스 텐서 [2, num_edges]
    
    Returns:
        torch.Tensor: 각 엣지의 특징 벡터 [num_edges, 4]
                      각 벡터는 [dx, dy, dz, distance]
    """
    
    # edge_index를 이용하여 각 엣지의 시작 노드와 종료 노드의 특징을 추출
    node1 = x[edge_index[0, :]]  
    node2 = x[edge_index[1, :]]  
    
    # 시작 노드와 종료 노드 간의 좌표 차이를 계산
    dxyz = node1 - node2  
    
    # 각 엣지에 대해 L2 norm(유클리드 거리)를 계산
    distance = torch.norm(dxyz, dim=1).unsqueeze(-1)
    
    edge_features = torch.cat([dxyz, distance], dim=-1) # [M, 4]
    
    return edge_features



def rigid_body_transform_graph(input_graph, euler_angles, translation):
    """
    Takes the input graph and applies a rigid body transformation to it.

    input: torch_geometric.data.Data
        The input graph data, encoded in the form described in train.py
    
    
    """
    # Extract the graph information
    x = input_graph.x
    pos = x[:, :3] # position (N,3)
    curv = x[:, 3:6] # curvature (N,3)
    norm = x[:, 6:9] # normal (N,3)
    bc = input_graph.bc # BC (N,2)
    y = input_graph.y # y (N,1)
    edge_index = input_graph.edge_index # edge_index (2,M)
    # transform y if necessary later on

    # Calculate the rotation matrix
    rot = Rotation.from_euler('zyx', euler_angles, degrees=True)
    rot_matrix = rot.as_matrix()

    # Perform the rigid body transformation
    pos = torch.matmul(rot_matrix, pos.T).T + translation
    curv = torch.matmul(rot_matrix, curv.T).T
    norm = torch.matmul(rot_matrix, norm.T).T

    # Create the resulting graph
    x = torch.cat([pos, curv, norm], dim=-1)
    result_graph = Data(x = x, bc = bc, edge_index = edge_index, y = y)
    return result_graph

def encode_for_goten(input_graph):
    """
    Takes the input graph and calculates the atomic_numbers, edge_diff, and edge_vec

    Input: torch_geometric.data
        The input graph data

    Output: 
        atomic_numbers: encoding the boundary conditions, (num_nodes)
        edge_diff: stores the absolute value of the edge vector (num_edges, 1)
        edge_vec: stores the edge vectors (num_edges, 3)
    """

    pos = input_graph.x[:, :3] # Position, (objects, 3)
    source, destination = input_graph.edge_index # (2, num_edges)
    atomic_numbers = input_graph.bc

    edge_vec = pos[destination] - pos[source] # (num_edges, 3)
    edge_diff = torch.linalg.norm(edge_vec, ord = 2, dim = 1, keepdim = True) # (num_edges, 1)

    return atomic_numbers, edge_diff, edge_vec