import torch.nn as nn



#-------------------#
# Config

class LIMoEConfig:
    def __init__(
        self, 
        vocab_size,
        num_experts, 
        num_tasks, 
        moe_input_size,
        moe_hidden_size,
        moe_output_size,
        top_k=4,
        noisy_gating=True,
        max_position_embeddings=512,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        hidden_dim=768, 
        num_layers=8, 
        dropout=0.1, 
        hidden_dropout_prob=0.1, 
        pre_lnorm=True,
        n_heads=8,
        image_size=224,
        patch_size=16,
        num_channels=3,
        max_image_length=-1,
        layer_norm_eps=1e-5,
        expert_activation=nn.ReLU(), 
        task_activation=nn.ReLU(), 
        output_activation=nn.Sigmoid(),
        hidden_act="gelu",
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        is_decoder=False,
    ):
        # Input
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size

        # MoE
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.moe_input_size = moe_input_size
        self.moe_hidden_size = moe_hidden_size
        self.moe_output_size = moe_output_size

        # image
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_image_length = max_image_length

        # Transformer
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pre_lnorm = pre_lnorm
        self.n_heads = n_heads
        self.d_heads = int(hidden_dim / n_heads)

        # LayerNorm
        self.layer_norm_eps = layer_norm_eps

        # Activations
        self.expert_activation = expert_activation
        self.task_activation = task_activation
        self.output_activation = output_activation
        self.hidden_act = hidden_act

        # Other
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.is_decoder = is_decoder
    

    def set_input_params(self, vocab_size, hidden_size, type_vocab_size=2, modality_type_vocab_size=2):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size
    
    def set_moe_params(
        self, 
        num_experts, 
        num_tasks, 
        moe_input_size,
        moe_hidden_size,
        moe_output_size,
        top_k=4,
        noisy_gating=True,
    ):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.moe_input_size = moe_input_size
        self.moe_hidden_size = moe_hidden_size
        self.moe_output_size = moe_output_size

    def set_transformer_params(
        self, 
        hidden_dim, 
        num_layers, 
        dropout, 
        pre_lnorm=True,
        n_heads=8,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.n_heads = n_heads
        self.d_heads = int(hidden_dim / n_heads)
        self.hidden_dropout_prob = hidden_dropout_prob

    def set_layer_norm_params(self, layer_norm_eps=1e-5):
        self.layer_norm_eps = layer_norm_eps

    def set_image_params(self, image_size=224, patch_size=16, num_channels=3, max_image_length=-1):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_image_length = max_image_length
