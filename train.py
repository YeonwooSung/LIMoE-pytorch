import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import ViltFeatureExtractor
import pandas as pd
import os
from PIL import Image

from limoe import LIMoE, LIMoEConfig, MLMHead, ITMHead, compute_mlm, compute_itm


TOKENIZER_FILE = "tokenizer.json"


class TrainDataset(Dataset):
    def __init__(self, tokenizer, feature_extractor, data):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image"]
        text = self.data[idx]["text"]
        mlm_label = self.data[idx]["mlm_label"]
        itm_label = self.data[idx]["itm_label"]

        # Load image
        image = Image.open(image_path)

        # Encode text
        encoded_text = self.tokenizer.encode(text)
        input_ids = encoded_text.ids
        attention_mask = encoded_text.attention_mask

        # Encode image
        encoded_image = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoded_image.pixel_values
        pixel_mask = encoded_image.pixel_mask

        mlm_label = torch.tensor(mlm_label)
        itm_label = torch.tensor(itm_label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "mlm_label": mlm_label,
            "itm_label": itm_label,
        }


def load_data(data_file):
    data = pd.read_csv(data_file)
    return data


def train_limoe(model, mlm_head, itm_head, optimizer, data_loader, target_device="cuda"):
    device = torch.device(target_device)
    model.train()
    model.to(device)

    # Train model
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        mlm_label = batch["mlm_label"].to(device)
        itm_label = batch["itm_label"].to(device)

        
        # Forward pass
        logits = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            pixel_mask=pixel_mask
        )
        mlm_logits = mlm_head(logits)
        itm_logits = itm_head(logits)

        # Compute loss
        mlm_loss = compute_mlm(mlm_logits, mlm_label, model.config.vocab_size)
        itm_loss = compute_itm(itm_logits, itm_label)

        # Backward pass
        loss = mlm_loss / 2 + itm_loss / 2
        loss.backward()
        optimizer.step()


def main(config, max_length=128, batch_size=32, epochs=10, learning_rate=1e-4):
    # Load tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=max_length)

    # Load feature extractor
    feature_extractor = ViltFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    # instantiate LIMoE model
    model = LIMoE(config)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataset
    data = load_data("data.csv")
    dataset = TrainDataset(tokenizer, feature_extractor, data)

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # head models
    mlm_head = MLMHead(config)
    itm_head = ITMHead(config)

    # Train model
    for epoch in range(epochs):
        train_limoe(model, mlm_head, itm_head, tokenizer, feature_extractor, optimizer, data_loader)


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

    main(config, max_length=128, batch_size=32, epochs=10, learning_rate=1e-4)
