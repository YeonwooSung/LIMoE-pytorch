from turtle import forward
import torch
import torch.nn as nn


#-------------------#

class LIMoEConfig:
    def __init__(
        self, 
        input_dim, 
        num_experts, 
        num_tasks, 
        hidden_dim=1024, 
        num_layers=8, 
        dropout=0.1, 
        expert_activation=nn.ReLU(), 
        task_activation=nn.ReLU(), 
        output_activation=nn.Sigmoid()
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.expert_activation = expert_activation
        self.task_activation = task_activation
        self.output_activation = output_activation

#-------------------#
# Mixture of Experts

class MoE(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()

#-------------------#
# Sparse Attention block

class SparseSelfAttentionBlock(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()

    def forward(self, x):
        return x

#-------------------#
# Dense Attention

class DenseSelfAttentionBlock(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()

    def forward(self, x):
        return x

#-------------------#
# LIMoE Encoder Layer

class LIMoEEncoderLayer(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.dense_attention = DenseSelfAttentionBlock(config)
        self.sparse_attention = SparseSelfAttentionBlock(config)

    def forward(self, x):
        return x

#-------------------#
# LIMoE Encoder

class LIMoEEncoder(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList([LIMoEEncoderLayer(config) for _ in range(self.num_layers)])

    def forward(self, x):
        pass


#-------------------#
# LIMoE model

class LIMoE(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x):
        pass
