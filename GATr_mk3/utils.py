import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader
from gatr.interface import (
    embed_point,
    embed_scalar,
    embed_pluecker_ray,
    embed_oriented_plane,
    embed_translation,
    extract_scalar,
    extract_point,
    extract_pluecker_ray,
    extract_oriented_plane,
    extract_point_embedding_reg,
)

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


def encode_to_PGA(graph_data):
    """
    Takes the graph data and encodes it into PGA data.

    The encoding is as follows:
    Boundary condition: encoded as a scalar
        (Originally, the BC was encoded in two dimensions but we can just encode it as a single scalar by using the number -1 as well)
    Node position: encoded as a trivector
    Node curvature: encoded as a bivector, with the heading being the curvature and the position being the position of the node
    Node normal: encoded as a vector, with the position info also encoded in the vector

    Parameters
    ----------
    inputs: torch_geometric.data.Data
        The graph data to be encoded
        (Orignal GATr considered batchsize, but here we only load one graph at a time)

    returns
    -------
    multivector: torch.Tensor with shape (batchsize, objects, 1, 16)
        PGA embeddings
    scalar: torch.Tensor with shape (batchsize, objects, 1, scalar_size)
        scalar embeddings
    """

    # Extract the data from the graph
    n_objects = graph_data.x.shape[0] # Number of objects
    pos = graph_data.x[:, :3] # Position, (objects, 3)
    curv = graph_data.x[:, 3:6] # Curvature, (objects, 3)
    norm = graph_data.x[:, 6:9] # Normal, (objects, 3) 
    bc = graph_data.bc # BC (objects, 2)
    source, destination = graph_data.edge_index # (2, num_edges)

    # Calculate the edge vectors and turn them into edge features
    edge_vectors = pos[destination] - pos[source] # (num_edges, 3)
    node_edge_v = torch.zeros(n_objects, 3).to(graph_data.x.device)
    for index in range(n_objects):
        out_mask = (source == index)
        in_mask = (destination == index)
        out_edge_vectors = edge_vectors[out_mask]
        in_edge_vectors = -edge_vectors[in_mask]
        node_edge_vectors = torch.cat([out_edge_vectors, in_edge_vectors], dim=0) # (num_edges from that node, 3)
        node_edge_vectors = node_edge_vectors / ((torch.linalg.norm(node_edge_vectors, dim = 1, keepdim = True))**2)
        node_edge_v[index, :] = node_edge_vectors.sum(dim = 0)


    # Embed the data into the PGA
    pos_PGA = embed_point(pos) # Embed the position into a trivector, (objects, 16)
    # curv_PGA= embed_pluecker_ray(torch.cat([curv, torch.linalg.cross(pos, curv)], dim = 1)) # Embed the curvature into a bivector, (objects, 16)
    # norm_PGA = embed_oriented_plane(norm, pos) # Embed the normal into a vector, (objects, 16)
    node_edge_PGA = embed_oriented_plane(node_edge_v, pos)

    
    # Initially planned on encoding BC in the scalar space, but since GATr has a seperate scalar flow, not necessary
    #bc_compressed = bc[:,0] - bc[:, 1] # Modified BC, rbe2 now encoded as 1 and rbe3 is now encoded as -1
    #bc_PGA = embed_scalar(bc_compressed.unsqueeze(1)) # Embed the BC into a scalar, (objects, 16)
    
    # Concatenate the embeddings
    multivector = pos_PGA + node_edge_PGA  # (objects, 16)
    # Set the scalar and pseudoscalar components to 0 since they aren't used here 
    # multivector[:,0] = 0
    # multivector[:, 15] = 0

    # Insert the channel and batch_size dimensions
    multivector = multivector.unsqueeze(0).unsqueeze(2) # (1, objects, 1, 16)
    scalar = bc.unsqueeze(0) # (1, objects, 2)

    return multivector, scalar

def extract_from_PGA(multivector, scalar, base_graph):
    """
    Takes the multivector and returns is back to the graph data form

    Note that the output multivectors from a neural network no longer hold any actual physical meaning
    and therefore those variable names should only be considered as temporary placeholders 

    Parameters
    ----------
    inputs: torch.Tensor with shape (1, objects, 1, 16)
                The multivector to be decoded
            torch.Tensor with shape (1, objects, 2)
                The scalar to be decoded
            base_graph: torch_geometric.data.Data
                The base graph data that contains edge information, ground truth etc.

    returns
    -------
    graph_data: torch_geometric.data.Data
        The graph data
    """

    # Remove the channel and batch_size dimensions
    multivector = multivector.squeeze(0).squeeze(1)
    scalar = scalar.squeeze(0)

    # Extract the data from the multivector
    pos = extract_point(multivector) # Extract the position from the trivector, (objects, 3)
    curv = extract_pluecker_ray(multivector) # Extract the curvature from the bivector, (objects, 6)
    curv = curv[:, :3] # The last three elements are the position, so we remove them
    norm = extract_oriented_plane(multivector) # Extract the normal from the vector, (objects, 3)
    #bc = scalar
    bc = extract_scalar(multivector) # Extract the scalar component, which for now is the predicted stress (objects, 1)

    # BC not encoded in multivector anymore
    # bc = extract_scalar(multivector) # Extract the BC from the scalar, (objects, 1)
    # bc = torch.cat([torch.nn.functional.relu(bc), torch.nn.functional.relu(-bc)], dim=1) # Decode the BC back to the original form, (objects, 2)

    x = torch.cat([pos, curv, norm], dim=-1) # (objects, 9)

    result_graph = Data(x = x, bc = bc, edge_index = base_graph.edge_index, y = base_graph.y)

    return result_graph

def von_mises(stress):
    """
    Takes the principal stress values and returns the von Mises stress

    Input: torch.Tensor with shape (objects, 3)
        The principal stress values

    Output: torch.Tensor with shape (objects, 1)
        The von Mises stress values
    """
    # extract each principle stress
    sigma1 = stress[:, 0]
    sigma2 = stress[:, 1]
    sigma3 = stress[:, 2]

    # calculate the von Mises stress
    von_mises = torch.sqrt(0.5*((sigma1-sigma2)**2 + (sigma2-sigma3)**2 + (sigma3-sigma1)**2))
    return von_mises.unsqueeze(1)


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
