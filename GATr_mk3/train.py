import os 
import numpy as np
import wandb
import hydra
import h5py

from torchinfo import summary
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.utils import to_undirected, add_self_loops

from modules import *
from utils import *

class Trainer():
    def __init__(self, cfg: DictConfig, shuffle=True):
        self.device = cfg.experiment.device
        self.max_epochs = cfg.dataset.max_epochs

        self.ckpt_path = "ckpt"

        self.min_force = torch.tensor([-37809.9],dtype = torch.float32)
        self.max_force = torch.tensor([35585.8],dtype = torch.float32)
                
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        train_graph_list, valid_graph_list = self.process_data(cfg)

        self.train_loader = GraphLoader(train_graph_list, batch_size=1, shuffle=shuffle, pin_memory = True, num_workers = 4) # default는 True
        self.valid_loader = GraphLoader(valid_graph_list, batch_size=1, shuffle=False, pin_memory = True, num_workers = 4)  

        #self.make_model_components(cfg)
        
        self.min_train_loss = 1e+3
        self.min_val_loss = 1e+3

        self.min_train_epoch = 0
        self.min_val_epoch = 0


    def process_data(self, cfg):
        train_graph_list = []
        valid_graph_list = []
        targets_for_norm = []
        curvs_for_norm = []
        normals_for_norm = []
        coords_for_norm = []

        data_dir = cfg.dataset.data_dir

        train_sample_id = cfg.dataset.train_sample_id
        valid_sample_id = cfg.dataset.valid_sample_id

        for sample_idx in train_sample_id:
            file_name = f"small_ver_25/preprocessed/sample_{sample_idx}.h5"
            file_path = os.path.join(data_dir, file_name)

            with h5py.File(file_path, 'r') as file:

                coords = torch.tensor(file['coords'][()], dtype=torch.float32)          # [N, 3]
                edge_index = torch.tensor(file['edges'][()], dtype=torch.long)          # [2, N_edge]
                curv = torch.tensor(file['curvatures'][()], dtype=torch.float32)        # [N, 3]
                norm = torch.tensor(file['norm_vectors'][()], dtype=torch.float32)      # [N, 3]
                targets = torch.tensor(file['outputs'][:, 0, -1:], dtype=torch.float32) # [N,] 수직 실험의 stress
                force = torch.tensor(file['forces'][0, :], dtype=torch.float32)         # [3]
                rbe2 = torch.tensor(file['rbe2'][()], dtype=torch.long)                 # [N_rbe2]
                rbe3 = torch.tensor(file['rbe3'][()], dtype=torch.long)                 # [N_rbe3]

                N = coords.shape[0] # 노드 개수

                edge_index = to_undirected(edge_index)
                # edge_index = add_self_loops(edge_index)[0]

                coords = coords - torch.mean(coords, dim=0) # 중심점으로 이동

                rbe2_onehot = one_hot_from_rbe(rbe2, N)
                rbe3_onehot = one_hot_from_rbe(rbe3, N)

                force = (force - self.min_force) / (self.max_force - self.min_force)
                force = force.expand(N, -1)  # [N, 3]으로 확장
                # TODO: Encode the force information in to the graph

                x = torch.cat([coords, curv, norm], dim=-1) # [N, 9]
                bc = torch.cat([rbe2_onehot, rbe3_onehot], dim = -1) # [N, 2]
                # bc = torch.cat([force, rbe2_onehot, rbe3_onehot], dim=-1) # [N, 5]

                print(f"[train] sample_idx: {sample_idx:>2} | shape of input: {x.shape} | shape of bc: {bc.shape}")

                train_graph_list.append(Data(x = x, 
                                             bc = bc, 
                                             edge_index = edge_index, 
                                             y = targets))

                targets_for_norm.append(targets)
                curvs_for_norm.append(curv)
                normals_for_norm.append(norm)
                coords_for_norm.append(coords)


        for sample_idx in valid_sample_id:
            file_name = f"small_ver_25/preprocessed/sample_{sample_idx}.h5"
            file_path = os.path.join(data_dir, file_name)

            with h5py.File(file_path, 'r') as file:

                coords = torch.tensor(file['coords'][()], dtype=torch.float32)          # [N, 3]
                edge_index = torch.tensor(file['edges'][()], dtype=torch.long)          # [2, N_edge]
                curv = torch.tensor(file['curvatures'][()], dtype=torch.float32)        # [N, 3]
                norm = torch.tensor(file['norm_vectors'][()], dtype=torch.float32)      # [N, 3]
                targets = torch.tensor(file['outputs'][:, 0, -1:], dtype=torch.float32) # [N,] 수직 실험의 stress
                force = torch.tensor(file['forces'][0, :], dtype=torch.float32)         # [3]
                rbe2 = torch.tensor(file['rbe2'][()], dtype=torch.long)                 # [N_rbe2]
                rbe3 = torch.tensor(file['rbe3'][()], dtype=torch.long)                 # [N_rbe3]

                N = coords.shape[0] # 노드 개수

                edge_index = to_undirected(edge_index)
                # edge_index = add_self_loops(edge_index)[0]

                coords = coords - torch.mean(coords, dim=0) # 중심점으로 이동

                rbe2_onehot = one_hot_from_rbe(rbe2, N)
                rbe3_onehot = one_hot_from_rbe(rbe3, N)

                force = (force - self.min_force) / (self.max_force - self.min_force)
                force = force.expand(N, -1)  # [N, 3]으로 확장

                x = torch.cat([coords, curv, norm], dim=-1) # [N, 9]
                bc = torch.cat([rbe2_onehot, rbe3_onehot], dim = -1) # [N, 2]
                # bc = torch.cat([force, rbe2_onehot, rbe3_onehot], dim=-1) # [N, 5]

                print(f"[valid] sample_idx: {sample_idx:>2} | shape of input: {x.shape} | shape of bc: {bc.shape}")

                valid_graph_list.append(Data(x = x, 
                                             bc = bc, 
                                             edge_index = edge_index, 
                                             y = targets))


        print('# of train dataset:', len(train_graph_list))
        print('# of valid dataset', len(valid_graph_list))

        coords_train = torch.cat(coords_for_norm)
        curvs_train = torch.cat(curvs_for_norm)
        norms_train = torch.cat(normals_for_norm)

        targets_train = torch.cat(targets_for_norm)

        self.coords_max = torch.max(coords_train)
        self.coords_min = torch.min(coords_train)
        
        self.curv_mean = torch.mean(curvs_train)
        self.curv_std = torch.std(curvs_train)

        self.norm_mean = torch.mean(norms_train)
        self.norm_std = torch.std(norms_train)

        self.target_mean = torch.mean(targets_train)
        self.target_std = torch.std(targets_train)

        # normalization
        for data in train_graph_list:
            data.y = (data.y - self.target_mean) / self.target_std

            data.x[:, :3] = (data.x[:, :3] - self.coords_min) / (self.coords_max - self.coords_min)
            data.x[:, 3:6] = (data.x[:, 3:6] - self.curv_mean) / self.curv_std
            #data.x[:, 6:9] = (data.x[:, 6:9] - self.norm_mean) / self.norm_std
            data.x[:, 6:9] = nn.functional.normalize(data.x[:, 6:9], p=2.0, dim = -1) # I have been told this is already normalized

            # data.bc = (data.bc - self.target_mean) / self.target_std

            data.edge_weight = calculate_edge_features(data.x[:, 0:3], data.edge_index)

        for data in valid_graph_list:
            data.y = (data.y - self.target_mean) / self.target_std

            data.x[:, :3] = (data.x[:, :3] - self.coords_min) / (self.coords_max - self.coords_min)
            data.x[:, 3:6] = (data.x[:, 3:6] - self.curv_mean) / self.curv_std
            # data.x[:, 6:9] = (data.x[:, 6:9] - self.norm_mean) / self.norm_std
            data.x[:, 6:9] = nn.functional.normalize(data.x[:, 6:9], p=2.0, dim = -1) # I have been told this is already normalized

            # data.bc = (data.bc - self.target_mean) / self.target_std

            data.edge_weight = calculate_edge_features(data.x[:, 0:3], data.edge_index)

        self.input_dim = x.shape[1]
        self.bc_dim = bc.shape[1]
        self.output_dim = targets.shape[1]

        return train_graph_list, valid_graph_list


   
    def make_model_components(self, cfg: DictConfig):

        self.model = GATrmk3(
            in_mv_channel = cfg.arch.encoder.in_mv_channel,
            in_s_channel = cfg.arch.encoder.in_s_channel,
            hidden_mv_channel= cfg.arch.processor.hidden_mv_channel,
            hidden_s_channel = cfg.arch.processor.hidden_s_channel,
            n_attn_heads= cfg.arch.processor.n_attn_heads,
            n_mlp_per_gatr = cfg.arch.processor.n_mlp_per_gatr,
            n_layers_gatr = cfg.arch.processor.n_layers_gatr,
            out_mv_channel = cfg.arch.decoder.out_mv_channel,
            out_s_channel = cfg.arch.decoder.out_s_channel
         ).to(self.device)

        summary(self.model)

        self.criterion = nn.MSELoss()
        #self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.scheduler.initial_lr, weight_decay = cfg.scheduler.weight_decay)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epochs)


    def train(self, cfg):
        for epoch in range(self.max_epochs):
            train_loss = 0
            self.model.train()
            
            for graph in self.train_loader:
                self.optimizer.zero_grad()
                graph = graph.to(self.device, non_blocking = True)
                # result_graph = self.model(graph)
                # pred = result_graph.bc.squeeze(-1)
                pred = self.model(graph)
                pred = pred.squeeze(-1)
                y = graph.y.flatten()
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                # graph.to("cpu").detach()
                
            train_loss /= len(self.train_loader)
            self.train_save_ckpt(train_loss, epoch)
            valid_loss = self.val()
            self.valid_save_ckpt(valid_loss, epoch)
            
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:>4}/{self.max_epochs:>4}] | Train Loss: {train_loss:.6f} | Val Loss: {valid_loss:.6f} | LR: {lr:.0e}")
            # self.scheduler.step()
            
            if cfg.experiment.wandb:
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})

        print(f"Best Train Loss: {self.min_train_loss:.6f} at epoch {self.min_train_epoch+1}")
        print(f"Best Valid Loss: {self.min_val_loss:.6f} at epoch {self.min_val_epoch+1}")

    @torch.no_grad()
    def val(self):
        valid_loss = 0
        self.model.eval()
        for graph in self.valid_loader:
            graph = graph.to(self.device, non_blocking = True)
            # result_graph = self.model(graph)
            # pred = result_graph.bc.squeeze(-1)
            pred = self.model(graph)
            pred = pred.squeeze(-1)
            y = graph.y.flatten()
            loss = self.criterion(pred, y)
            valid_loss += loss.item()

        valid_loss /= len(self.valid_loader)
        return valid_loss


    def train_save_ckpt(self, loss, epoch):
        if loss < self.min_train_loss:
            self.min_train_loss = loss
            self.min_train_epoch = epoch
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "best_train_loss.pt"))


    def valid_save_ckpt(self, loss, epoch):
        if loss < self.min_val_loss:
            self.min_val_loss = loss
            self.min_val_epoch = epoch            
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "best_valid_loss.pt"))

    def load_model(self, ckpt):
        self.model.load_state_dict(ckpt)


@hydra.main(config_path="config", version_base="1.1", config_name="config.yaml") 

def main(cfg: DictConfig):
    ###### 시드 고정 ######
    seed = cfg.experiment.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ###### weight and bias 설정 ######
    if cfg.experiment.wandb:
        os.environ['WANDB_API_KEY'] = cfg.experiment.wandb_api_key
        wandb.init(project = cfg.experiment.wandb_project_name)

    ###### 학습 및 평가 시작 ######
    trainer = Trainer(cfg)
    trainer.train(cfg)

if __name__  == "__main__":
    main()