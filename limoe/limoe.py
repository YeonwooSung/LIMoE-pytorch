import torch
import torch.nn as nn

from .config import LIMoEConfig
from .moe import MoE
from .helper import StableDropout


#-------------------#
# LIMoE Layer Normalization

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
# LIMoE Embedding

class LIMoETextEmbedding(nn.Module):
    def __init__(self, config:LIMoEConfig) -> None:
        super().__init__()

        pad_token_id = getattr(config, "pad_token_id", 0)
        hidden_size = config.hidden_size

        # word embeddings
        self.embedding_size = getattr(config, "embedding_size", hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        # position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        # token type embeddings
        type_vocab_size = getattr(config, "type_vocab_size", 2)
        if type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, self.embedding_size)

        # projection layer
        if self.embedding_size != hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, hidden_size, bias=False)
        
        self.LayerNorm = LIMoELayerNorm(hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        if hasattr(self, "token_type_embeddings"):
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if hasattr(self, "embed_proj"):
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



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

        self.embeddings = LIMoETextEmbedding(config)
        # self.embeddings = LIMoEImageEmbedding(config)

        self.encoder = LIMoEEncoder(config)

        # TODO: add the output layer

    def forward(self, x):
        pass
