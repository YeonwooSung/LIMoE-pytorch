import torch.nn as nn



#-------------------#
# Config

class LIMoEConfig:
    def __init__(
        self, 
        input_dim, 
        num_experts, 
        num_tasks, 
        moe_input_size,
        moe_hidden_size,
        moe_output_size,
        top_k=4,
        noisy_gating=True,
        hidden_dim=1024, 
        num_layers=8, 
        dropout=0.1, 
        pre_lnorm=True,
        n_heads=8,
        d_heads=64, #TODO need to check d_heads in the paper
        layer_norm_eps=1e-5,
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
        self.moe_output_size = moe_output_size

        # Transformer
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.n_heads = n_heads
        self.d_heads = d_heads

        # LayerNorm
        self.layer_norm_eps = layer_norm_eps

        # Activations
        self.expert_activation = expert_activation
        self.task_activation = task_activation
        self.output_activation = output_activation