import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import ViltFeatureExtractor

from limoe import LIMoE, LIMoEForImageAndTextRetrieval, LIMoEConfig


TOKENIZER_FILE = "tokenizer.json"


class TrainDataset(Dataset):
    def __init__(self, tokenizer, feature_extractor, data):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]

        # Encode text
        encoded_text = self.tokenizer.encode(text)
        input_ids = encoded_text.ids
        attention_mask = encoded_text.attention_mask

        # Encode image
        encoded_image = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoded_image.pixel_values
        pixel_mask = encoded_image.pixel_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": label,
        }


def train_limoe(model, optimizer, loss_fn, data_loader, target_device="cuda"):
    device = torch.device(target_device)
    model.train()
    model.to(device)

    # Train model
    for batch in data_loader:
        optimizer.zero_grad()
        loss = loss_fn(model, batch) #TODO
        loss.backward()
        optimizer.step()


def main(max_length=128, batch_size=32, epochs=10, learning_rate=1e-4):
    # Load tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=max_length)

    # Load feature extractor
    feature_extractor = ViltFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Create model
    config = LIMoEConfig(
        num_classes=1000,
        num_experts=8,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=128,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        gradient_checkpointing=False,
        use_cache=False,
    )
    model = LIMoEForImageAndTextRetrieval(config)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create loss function
    loss_fn = nn.CrossEntropyLoss()

    # Create dataset
    data = None
    dataset = TrainDataset(tokenizer, feature_extractor, data)

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train model
    for epoch in range(epochs):
        train_limoe(model, tokenizer, feature_extractor, optimizer, loss_fn, data_loader)


if __name__ == "__main__":
    num_experts, num_tasks = 4, 2
    moe_input_size, moe_hidden_size, moe_output_size = 768, 512, 768

    # load tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    vocab_size = tokenizer.get_vocab_size()

    # load feature extractor
    feature_extractor = ViltFeatureExtractor(size=384)

    # generate LIMoE config
    config = LIMoEConfig(
        vocab_size,
        num_experts,
        num_tasks,
        moe_input_size,
        moe_hidden_size,
        moe_output_size,
    )

    # instantiate LIMoE model
    model = LIMoE(config)

    text_encodings = tokenizer.encode("Hello world!")
    #TODO
