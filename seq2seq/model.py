"""Seq2Seq model architecture module."""

import torch
import torch.nn as nn


class SimpleSeq2Seq(nn.Module):
    """Simple sequence-to-sequence model using GRU."""

    def __init__(self, vocab_size: int, hidden_size: int):
        """
        Initialize the Seq2Seq model.

        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Size of the hidden state in GRU
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, char_idx: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            char_idx: Input character indices [seq_len, batch]
            hidden: Hidden state from previous step

        Returns:
            tuple: (output predictions, hidden state)
        """
        emb = self.embedding(char_idx)  # [seq_len, batch, hidden_size]
        output, hidden = self.gru(emb, hidden)
        prediction = self.out(output)  # [seq_len, batch, vocab_size]
        return prediction, hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """
        Initialize hidden state with zeros.

        Args:
            batch_size: Batch size for the hidden state

        Returns:
            Zero-initialized hidden state
        """
        return torch.zeros(1, batch_size, self.hidden_size)
