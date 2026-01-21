"""Training module for Seq2Seq model."""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Callable, Optional

from .config import Config
from .model import SimpleSeq2Seq
from .data import Dataset


class Trainer:
    """Trainer class for Seq2Seq model."""

    def __init__(
        self,
        model: SimpleSeq2Seq,
        dataset: Dataset,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        learning_rate: float = Config.LEARNING_RATE,
    ):
        """
        Initialize the trainer.

        Args:
            model: The Seq2Seq model to train
            dataset: Dataset object containing training data
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.dataset = dataset
        self.device = Config.get_device()

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=learning_rate)

    def train_epoch(self) -> dict:
        """
        Train for one epoch.

        Returns:
            Dictionary containing loss, accuracy, and perplexity metrics
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_chars = 0

        for company, name in self.dataset.get_training_pairs():
            hidden = self.model.init_hidden(1).to(self.device)

            # Vectorized input: whole string at once
            full_seq = company + Config.SEPARATOR_TOKEN + name
            tensor_seq = self.dataset.to_tensor(full_seq)

            input_data = tensor_seq[:-1]
            target_labels = tensor_seq[1:].view(-1)

            self.optimizer.zero_grad()
            output, hidden = self.model(input_data, hidden)
            output_flat = output.view(-1, self.dataset.vocabulary.vocab_size)

            loss = self.criterion(output_flat, target_labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predictions = output_flat.argmax(dim=1)
            total_correct += (predictions == target_labels).sum().item()
            total_chars += target_labels.size(0)

        avg_loss = total_loss / len(self.dataset)
        accuracy = total_correct / total_chars
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
        }

    def train(
        self,
        epochs: int = Config.EPOCHS,
        print_every: int = Config.PRINT_EVERY,
        eval_callback: Optional[Callable] = None,
    ) -> None:
        """
        Train the model for multiple epochs.

        Args:
            epochs: Number of epochs to train
            print_every: Print metrics every N epochs
            eval_callback: Optional callback function for evaluation during training
        """
        print(f"Using device: {self.device}")

        pbar = tqdm(range(1, epochs + 1), desc="Training")
        for epoch in pbar:
            metrics = self.train_epoch()

            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                acc=f"{metrics['accuracy']:.2%}",
                perp=f"{metrics['perplexity']:.2f}",
            )

            if epoch % print_every == 0 and eval_callback:
                eval_callback(epoch, metrics)

    def save_model(self, path: str = "seq2seq_model.pth") -> None:
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
        print(f"\nModel saved to {path}")

    def load_model(self, path: str = "seq2seq_model.pth") -> None:
        """
        Load a trained model.

        Args:
            path: Path to load the model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
