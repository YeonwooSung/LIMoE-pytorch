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
        moe_input_size,
        moe_hidden_size,
        top_k=4,
        noisy_gating=True,
        hidden_dim=1024, 
        num_layers=8, 
        dropout=0.1, 
        expert_activation=nn.ReLU(), 
        task_activation=nn.ReLU(), 
        output_activation=nn.Sigmoid()
    ):
        # Input
        self.input_dim = input_dim

        # MoE
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.moe_input_size = moe_input_size
        self.moe_hidden_size = moe_hidden_size

        # Transformer
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Activations
        self.expert_activation = expert_activation
        self.task_activation = task_activation
        self.output_activation = output_activation


#-------------------#
# Mixture of Experts

class MoE(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.noisy_gating = config.noisy_gating
        self.k = config.top_k
        self.input_size = config.moe_input_size
        self.hidden_size = config.moe_hidden_size

        # add experts for MoE
        self.experts = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size) for _ in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

    def forward(self, x):
        pass

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
