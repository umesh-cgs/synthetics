import torch
import torch.nn as nn
import csv
from tqdm import tqdm
import math
import difflib

# HYPERPARAMETERS
epochs = 100
print_every = 1  # Show updates every epoch to diagnose hanging
hidden_size = 64  # Size of the "memory" of the RNN
learning_rate = 0.005


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Helpers
def to_tensor(s):
    return torch.tensor([char_to_i[c] for c in s], device=device).unsqueeze(1)


def int_to_char(i):
    return chars[i]


# Data Loading

with open("cmpnydta.csv", encoding="utf-8") as f:
    reader = csv.reader(f)

    next(reader)

    data = [tuple(row) for row in reader]

# Create a simple char mapping
chars = sorted(
    list(set("".join([a + b for a, b in data]) + ">"))
)  # '>' is our separator
char_to_i = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)


class SimpleSeq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, char_idx, hidden):
        # char_idx can be [seq_len, batch]
        emb = self.embedding(char_idx)  # [seq_len, batch, hidden_size]
        output, hidden = self.gru(emb, hidden)
        prediction = self.out(output)  # [seq_len, batch, vocab_size]
        return prediction, hidden


def evaluate(model, input_string, max_length=20):
    with torch.no_grad():
        hidden = torch.zeros(1, 1, hidden_size).to(device)
        input_tensor = to_tensor(input_string + ">")

        # 1. "Warm up" the hidden state with the input sequence at once
        output, hidden = model(input_tensor, hidden)

        # 2. Start generating characters
        result = ""
        # The last prediction from the sequence is the first char of the name
        top_v, top_i = output[-1].data.topk(1)

        for _ in range(max_length):
            char = int_to_char(top_i.item())
            if char == ">":
                break  # Stop if it predicts a separator (or add an <EOS> token)
            result += char

            # Feed the predicted character back in as the next input
            output, hidden = model(top_i.squeeze(), hidden)
            top_v, top_i = output.data.topk(1)

        return result


# TODO :  Extract this to a separate file
# Initialization steps
model = SimpleSeq2Seq(vocab_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)


pbar = tqdm(range(1, epochs + 1), desc="Training")
for epoch in pbar:
    total_loss = 0
    total_correct = 0
    total_chars = 0

    for company, name in data:
        hidden = torch.zeros(1, 1, hidden_size).to(device)

        # Vectorized input: whole string at once
        full_seq = company + ">" + name
        tensor_seq = to_tensor(full_seq)

        input_data = tensor_seq[:-1]
        target_labels = tensor_seq[1:].view(-1)

        opt.zero_grad()
        output, hidden = model(input_data, hidden)
        output_flat = output.view(-1, vocab_size)

        loss = criterion(output_flat, target_labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()

        # Calculate accuracy
        predictions = output_flat.argmax(dim=1)
        total_correct += (predictions == target_labels).sum().item()
        total_chars += target_labels.size(0)

    avg_loss = total_loss / len(data)
    accuracy = total_correct / total_chars
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    pbar.set_postfix(
        loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2%}", perp=f"{perplexity:.2f}"
    )

    # --- METRICS & EVALUATION ---
    if epoch % print_every == 0:
        # Visual check
        test_company, target_name = data[0]
        predicted_name = evaluate(model, test_company)

        # Calculate similarity ratio (0 to 1)
        similarity = difflib.SequenceMatcher(None, target_name, predicted_name).ratio()

        print(
            f"\nEpoch: {epoch} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | Perp: {perplexity:.2f}"
        )
        print(
            f"   Input: {test_company} -> Predicted: {predicted_name} (Target: {target_name}, Sim: {similarity:.2%})"
        )

# Save the model
torch.save(model.state_dict(), "seq2seq_model.pth")
print("\nModel saved to seq2seq_model.pth")
