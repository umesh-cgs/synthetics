"""Configuration module for Seq2Seq model training."""


class Config:
    """Configuration class containing all hyperparameters and settings."""

    # Training hyperparameters
    EPOCHS = 100
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.005
    PRINT_EVERY = 1

    # Data settings
    DATA_FILE = "cmpnydta.csv"
    SEPARATOR_TOKEN = ">"

    # Generation settings
    MAX_GENERATION_LENGTH = 20

    # Device settings
    DEVICE = "cuda"  # Will be set to 'cpu' if CUDA is unavailable

    @classmethod
    def get_device(cls):
        """Get the appropriate device (CUDA or CPU)."""
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
