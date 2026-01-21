"""Data loading and preprocessing module for Seq2Seq model."""

import csv
import torch
from typing import List, Tuple

from .config import Config


class Vocabulary:
    """Character-level vocabulary management."""

    def __init__(self, separator_token: str = Config.SEPARATOR_TOKEN):
        self.separator_token = separator_token
        self.chars = []
        self.char_to_i = {}
        self.i_to_char = {}
        self.vocab_size = 0

    def build_from_data(self, data: List[Tuple[str, str]]) -> None:
        """Build vocabulary from training data."""
        # Combine all company and name strings
        all_text = "".join([a + b for a, b in data])
        # Add separator token
        all_chars = list(set(all_text + self.separator_token))
        self.chars = sorted(all_chars)
        self.char_to_i = {c: i for i, c in enumerate(self.chars)}
        self.i_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> torch.Tensor:
        """Convert string to tensor of character indices."""
        indices = [self.char_to_i[c] for c in text]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """Convert tensor of indices back to string."""
        return "".join([self.i_to_char[i.item()] for i in indices])

    def char_at(self, index: int) -> str:
        """Get character at given index."""
        return self.i_to_char[index]


class Dataset:
    """Dataset loader and manager."""

    def __init__(self, data_file: str = Config.DATA_FILE):
        self.data_file = data_file
        self.data = []
        self.vocabulary = Vocabulary()
        self.device = Config.get_device()

    def load(self) -> List[Tuple[str, str]]:
        """Load data from CSV file."""
        with open(self.data_file, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.data = [tuple(row) for row in reader]
        return self.data

    def build_vocabulary(self) -> Vocabulary:
        """Build vocabulary from loaded data."""
        self.vocabulary.build_from_data(self.data)
        return self.vocabulary

    def to_tensor(self, s: str) -> torch.Tensor:
        """Convert string to tensor with proper shape."""
        indices = self.vocabulary.encode(s)
        return indices.unsqueeze(1).to(self.device)

    def get_training_pairs(self) -> List[Tuple[str, str]]:
        """Return all training pairs."""
        return self.data

    def __len__(self) -> int:
        return len(self.data)
