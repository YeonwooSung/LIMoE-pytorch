from turtle import forward
import torch
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
        top_k=4,
        noisy_gating=True,
        hidden_dim=1024, 
        num_layers=8, 
        dropout=0.1, 
        pre_lnorm=True,
        n_heads=8,
        d_heads=64, #TODO need to check d_heads in the paper
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
        self.pre_lnorm = pre_lnorm
        self.n_heads = n_heads
        self.d_heads = d_heads

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
# Self-Attention

class SelfAttention(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.pre_lnorm = config.pre_lnorm
        self.n_heads = config.n_heads
        self.d_heads = config.d_heads
        self.scale = 1 / (self.d_heads ** 0.5)
        qkv_output_dim = self.n_heads * self.d_heads

        # Q, K, V
        self.fc_q = nn.Linear(self.hidden_dim, qkv_output_dim)
        self.fc_k = nn.Linear(self.hidden_dim, qkv_output_dim)
        self.fc_v = nn.Linear(self.hidden_dim, qkv_output_dim)

        # Output
        self.fc_o = nn.Linear(qkv_output_dim, self.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(self.dropout)
        # LayerNorm
        self.ln = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        output_attentions=False,
    ):
        if query_states is None:
            query_states = hidden_states
        residual = hidden_states

        # LayerNorm
        if self.pre_lnorm:
            query_states = self.ln(query_states)
            kv_states = self.ln(hidden_states)
        else:
            kv_states = hidden_states

        # Q, K, V
        q = self.fc_q(query_states)
        k = self.fc_k(kv_states)
        v = self.fc_v(kv_states)

        # Split heads
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_heads).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_heads).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_heads).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = nn.Softmax(dim=-1)(attn)
        attn = self.dropout(attn)

        # Apply layer norm to the attention map for the Post-LayerNorm case
        if not self.pre_lnorm:
            attn = self.ln(attn)

        # Output
        attention_output = torch.matmul(attn, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(attention_output.size(0), attention_output.size(1), -1)
        attention_output = self.fc_o(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = attention_output + residual

        # Output
        outputs = (attention_output, attn) if output_attentions else attention_output
        return outputs

#-------------------#
# Sparse Attention block

class SparseSelfAttentionBlock(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.attention = SelfAttention(config)

        # MoE
        self.moe = MoE(config)

    def forward(self, x):
        return x

#-------------------#
# Dense Self-Attention Output

class DenseSelfAttentionOutputBlock(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.attention = SelfAttention(config)
    
    def forward(self, x):
        pass

#-------------------#
# Dense Attention

class DenseSelfAttentionBlock(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.attention = SelfAttention(config)
        self.outptut = DenseSelfAttentionOutputBlock(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        output_attentions=False,
    ):
        # perform the self-attention
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            query_states=query_states,
            output_attentions=output_attentions,
        )

        # pass the result of the self-attention layer to the dense ff-layer
        if output_attentions:
            attention_output, attn_matrix = attention_output
        attention_output = self.outptut(attention_output)

        # Output
        outputs = (attention_output, attn_matrix) if output_attentions else attention_output
        return outputs

#-------------------#
# LIMoE Encoder Layer

class LIMoEEncoderLayer(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.dense_attention = DenseSelfAttentionBlock(config)
        self.sparse_attention = SparseSelfAttentionBlock(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        pass

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
