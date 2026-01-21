"""Main entry point for Seq2Seq model training and inference."""

from .config import Config
from .data import Dataset
from .model import SimpleSeq2Seq
from .train import Trainer
from .evaluate import Evaluator


def main():
    """Main function to train the Seq2Seq model."""
    print("Initializing Seq2Seq model training...")

    # Load and prepare data
    print("Loading data...")
    dataset = Dataset()
    dataset.load()
    dataset.build_vocabulary()
    print(f"Loaded {len(dataset)} training pairs")
    print(f"Vocabulary size: {dataset.vocabulary.vocab_size}")

    # Initialize model
    print("\nInitializing model...")
    device = Config.get_device()
    model = SimpleSeq2Seq(
        vocab_size=dataset.vocabulary.vocab_size,
        hidden_size=Config.HIDDEN_SIZE,
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer and evaluator
    trainer = Trainer(model, dataset, learning_rate=Config.LEARNING_RATE)
    evaluator = Evaluator(model, dataset)

    # Define evaluation callback
    def eval_callback(epoch, metrics):
        evaluator.print_evaluation(epoch, metrics)

    # Train the model
    print("\nStarting training...")
    trainer.train(
        epochs=Config.EPOCHS,
        print_every=Config.PRINT_EVERY,
        eval_callback=eval_callback,
    )

    # Save the trained model
    trainer.save_model("seq2seq_model.pth")


def inference(input_string: str, model_path: str = "seq2seq_model.pth"):
    """
    Run inference on a single input string.

    Args:
        input_string: Input string to generate from
        model_path: Path to the saved model
    """
    print(f"\nRunning inference for: {input_string}")

    # Load data and vocabulary
    dataset = Dataset()
    dataset.load()
    dataset.build_vocabulary()

    # Initialize and load model
    device = Config.get_device()
    model = SimpleSeq2Seq(
        vocab_size=dataset.vocabulary.vocab_size,
        hidden_size=Config.HIDDEN_SIZE,
    ).to(device)

    trainer = Trainer(model, dataset)
    trainer.load_model(model_path)

    # Run inference
    evaluator = Evaluator(model, dataset)
    result = evaluator.evaluate_sample(input_string)

    print(f"Input: {result['input']}")
    print(f"Prediction: {result['prediction']}")


if __name__ == "__main__":
    main()
