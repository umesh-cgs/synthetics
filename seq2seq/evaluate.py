"""Evaluation and inference module for Seq2Seq model."""

import torch
import difflib
from typing import Optional

from .config import Config
from .model import SimpleSeq2Seq
from .data import Dataset


class Evaluator:
    """Evaluator class for Seq2Seq model inference."""

    def __init__(self, model: SimpleSeq2Seq, dataset: Dataset):
        """
        Initialize the evaluator.

        Args:
            model: The trained Seq2Seq model
            dataset: Dataset object containing vocabulary
        """
        self.model = model
        self.dataset = dataset
        self.device = Config.get_device()
        self.model.eval()

    def generate(
        self,
        input_string: str,
        max_length: int = Config.MAX_GENERATION_LENGTH,
    ) -> str:
        """
        Generate output for a given input string.

        Args:
            input_string: Input string (e.g., company name)
            max_length: Maximum length of generated output

        Returns:
            Generated output string
        """
        with torch.no_grad():
            hidden = self.model.init_hidden(1).to(self.device)
            input_tensor = self.dataset.to_tensor(input_string + Config.SEPARATOR_TOKEN)

            # 1. "Warm up" the hidden state with the input sequence at once
            output, hidden = self.model(input_tensor, hidden)

            # 2. Start generating characters
            result = ""
            # The last prediction from the sequence is the first char of the name
            top_v, top_i = output[-1].data.topk(1)

            for _ in range(max_length):
                char = self.dataset.vocabulary.char_at(top_i.item())
                if char == Config.SEPARATOR_TOKEN:
                    break
                result += char

                # Feed the predicted character back in as the next input
                # Ensure proper shape: [seq_len, batch] = [1, 1]
                next_input = top_i.view(-1, 1)
                output, hidden = self.model(next_input, hidden)
                top_v, top_i = output.data.topk(1)

            return result

    def evaluate_sample(self, input_string: str, target: Optional[str] = None) -> dict:
        """
        Evaluate a single sample.

        Args:
            input_string: Input string
            target: Optional target string for comparison

        Returns:
            Dictionary containing prediction and similarity metrics
        """
        prediction = self.generate(input_string)

        result = {
            "input": input_string,
            "prediction": prediction,
        }

        if target is not None:
            similarity = difflib.SequenceMatcher(None, target, prediction).ratio()
            result["target"] = target
            result["similarity"] = similarity

        return result

    def print_evaluation(self, epoch: int, metrics: dict) -> None:
        """
        Print evaluation results.

        Args:
            epoch: Current epoch number
            metrics: Training metrics dictionary
        """
        # Visual check: Let's see what it predicts for the first company
        test_company, target_name = self.dataset.get_training_pairs()[0]
        predicted_name = self.generate(test_company)

        # Calculate similarity ratio (0 to 1)
        similarity = difflib.SequenceMatcher(None, target_name, predicted_name).ratio()

        print(
            f"\nEpoch: {epoch} | Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.2%} | Perp: {metrics['perplexity']:.2f}"
        )
        print(
            f"   Input: {test_company} -> Predicted: {predicted_name} "
            f"(Target: {target_name}, Sim: {similarity:.2%})"
        )
