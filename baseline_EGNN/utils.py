import torch
import numpy as np
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
    rot = Rotation.from_euler('zyx', euler_angles.cpu().numpy(), degrees=True)
    rot_matrix = torch.from_numpy(rot.as_matrix()).float().to(pos.device)

    # Perform the rigid body transformation
    pos = torch.matmul(rot_matrix, (pos+translation).T).T
    curv = torch.matmul(rot_matrix, curv.T).T
    norm = torch.matmul(rot_matrix, norm.T).T

    # Create the resulting graph
    x = torch.cat([pos, curv, norm], dim=-1)
    result_graph = Data(x = x, bc = bc, edge_index = edge_index, y = y)
    return result_graph

def shuffle_nodes(data: Data) -> Data:
    """
    Randomly permutes the nodes in a PyTorch Geometric Data object,
    updating node-level attributes (like x, bc, and y) and the edge_index
    so that the graph structure (and positions) are preserved.

    Parameters
    ----------
    data : Data
        PyG data object containing:
        - x: node features (assumed shape [num_nodes, feature_dim])
        - edge_index: connectivity (shape [2, num_edges])
        - Optionally bc and y.

    Returns
    -------
    Data
        The updated data object with nodes permuted.
    """
    device = data.x.device
    num_nodes = data.x.size(0)
    
    # Generate a random permutation vector (new order) on the proper device.
    perm = torch.randperm(num_nodes, device=device)
    
    # Compute the inverse permutation mapping:
    # For each old index i, inverse_perm[i] = new index where node i appears.
    inverse_perm = torch.empty_like(perm, device=device)
    inverse_perm[perm] = torch.arange(num_nodes, device=device)
    
    # Permute node-level data:
    shuffled_x = data.x[perm]
    if hasattr(data, 'bc'):
        shuffled_bc = data.bc[perm]
    if hasattr(data, 'y'):
        shuffled_y = data.y[perm]
    
    # Ensure edge_index is on the target device
    shuffled_edge_index = data.edge_index.to(device)
    
    # Update edge_index: each value in edge_index is an old node index.
    # Replace them with their new index using the inverse permutation.
    shuffled_edge_index = inverse_perm[data.edge_index]

    shuffled_data = Data(x = shuffled_x, bc = shuffled_bc, y = shuffled_y, edge_index = shuffled_edge_index)
    
    return shuffled_data