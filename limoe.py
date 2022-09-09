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


#-------------------#
# Helper Classes

class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class LIMoELayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""
    def __init__(self, size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.float()
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_type)
        y = self.weight * hidden_states + self.bias
        return y


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

    def forward(self, hidden_states, input_tensor):
        #TODO
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

        # pass the result of the self-attention layer to the MoE sparse ff-layer
        if output_attentions:
            attention_output, attn_matrix = attention_output
        if query_states is None:
            query_states = hidden_states

        # MoE
        moe_output = self.moe(attention_output, query_states)

        # Output
        outputs = (moe_output, attn_matrix) if output_attentions else moe_output
        return outputs

#-------------------#
# Dense Self-Attention Output

class DenseSelfAttentionOutputBlock(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()
        self.LayerNorm = LIMoELayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

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
        if query_states is None:
            query_states = hidden_states
        attention_output = self.outptut(attention_output, query_states)

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
