import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import os

##############################################################################
# 1) ALU logic & dataset
##############################################################################
def alu_operation(a, b, opcode):
    if opcode == 0:      # 00 => AND
        return a & b
    elif opcode == 1:    # 01 => OR
        return a | b
    elif opcode == 2:    # 10 => ADD (mod 4)
        return (a + b) % 4
    else:                # 11 => XOR
        return a ^ b

def generate_dataset():
    inputs = []
    outputs = []
    for A in range(4):
        for B in range(4):
            for op in range(4):
                A_bin = format(A, '02b')
                B_bin = format(B, '02b')
                op_bin= format(op,'02b')

                A0, A1 = int(A_bin[1]), int(A_bin[0])
                B0, B1 = int(B_bin[1]), int(B_bin[0])
                Op0, Op1= int(op_bin[1]), int(op_bin[0])

                inp = [A0, A1, B0, B1, Op0, Op1]

                result = alu_operation(A, B, op)
                res_bin = format(result, '02b')
                Out0, Out1 = int(res_bin[1]), int(res_bin[0])

                inputs.append(inp)
                outputs.append([Out0, Out1])

    inputs_tensor  = torch.tensor(inputs,  dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
    return inputs_tensor, outputs_tensor


##############################################################################
# 2) Model Definition
##############################################################################
class ALUNet(nn.Module):
    def __init__(self, hidden_size=8):
        super(ALUNet, self).__init__()
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


##############################################################################
# 3) Normal Training & Evaluation
##############################################################################
def train_alu_model(device='cpu', epochs=2000):
    x_train, y_train = generate_dataset()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ALUNet(hidden_size=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss   = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"[Normal Train] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model

def evaluate_model(model, device='cpu'):
    x_test, y_test = generate_dataset()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        logits = model(x_test)
        preds = torch.sigmoid(logits)
        predicted_bits = (preds > 0.5).int()
        correct_bits   = (predicted_bits == y_test.int())
        accuracy = correct_bits.all(dim=1).float().mean().item()

    print(f"Accuracy on 64 patterns (no error): {accuracy*100:.2f}%")
    return accuracy


##############################################################################
# 4) Stuck-at-Zero TRAINING (Mask-based)
##############################################################################
def train_alu_model_stuck_at_zero(device='cpu', epochs=2000, stuck_neuron_indices=None):
    """
    Train the model while certain hidden neurons are "stuck at zero."
    We do this by multiplying the hidden activations with a mask that has
    0 for the stuck neurons, 1 for the functional ones.
    """
    if stuck_neuron_indices is None:
        stuck_neuron_indices = []

    x_train, y_train = generate_dataset()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ALUNet(hidden_size=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Build a neuron mask for shape [hidden_size].
    # e.g., if stuck_neuron_indices = [2,4], then mask = [1,1,0,1,0,1,1,1].
    neuron_mask = torch.ones(model.fc1.out_features, device=device)
    for idx in stuck_neuron_indices:
        neuron_mask[idx] = 0.0
    # We'll expand it to match batch dimension in each forward pass.

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass (manual)
        hidden = model.fc1(x_train)
        hidden = torch.relu(hidden)

        # Expand mask from [hidden_size] to [batch_size, hidden_size]
        batch_size = hidden.size(0)
        mask = neuron_mask.unsqueeze(0).expand(batch_size, -1)
        hidden = hidden * mask  # elementwise multiply, no in-place indexing

        logits = model.fc2(hidden)

        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"[Stuck-Train] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model

def evaluate_model_stuck_at_zero(model, device='cpu', stuck_neuron_indices=None):
    """
    Evaluate the model with certain neurons forced to zero in the hidden layer
    at inference time, using a similar mask approach (no in-place ops).
    """
    if stuck_neuron_indices is None:
        stuck_neuron_indices = []

    x_test, y_test = generate_dataset()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    neuron_mask = torch.ones(model.fc1.out_features, device=device)
    for idx in stuck_neuron_indices:
        neuron_mask[idx] = 0.0

    with torch.no_grad():
        hidden = model.fc1(x_test)
        hidden = torch.relu(hidden)

        batch_size = hidden.size(0)
        mask = neuron_mask.unsqueeze(0).expand(batch_size, -1)
        hidden = hidden * mask

        logits = model.fc2(hidden)
        preds = torch.sigmoid(logits)
        predicted_bits = (preds > 0.5).int()
        correct_bits   = (predicted_bits == y_test.int())
        accuracy = correct_bits.all(dim=1).float().mean().item()

    print(f"Stuck-at-Zero (neurons={stuck_neuron_indices}), Accuracy: {accuracy*100:.2f}%")
    return accuracy


##############################################################################
# 5) Logging to CSV
##############################################################################
CSV_FILENAME = "experiments_results.csv"

def append_result_to_csv(row_dict):
    file_exists = os.path.isfile(CSV_FILENAME)
    fieldnames = [
        "TestName",
        "RunIndex",
        "WhichNeuronsStuck",
        "OriginalAccuracy",
        "StuckTrainAccuracy"
    ]
    with open(CSV_FILENAME, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


##############################################################################
# 6) MAIN
##############################################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #------------------------------------------------------------------
    # 1) Train normal model once
    #------------------------------------------------------------------
    model_normal = train_alu_model(device=device, epochs=2000)
    normal_accuracy = evaluate_model(model_normal, device=device)

    #------------------------------------------------------------------
    # 2) Stuck-at-Zero Training: 25% random neurons x 5 runs
    #------------------------------------------------------------------
    STUCK_PERCENTAGE = 1
    HIDDEN_SIZE       = 8
    n_stuck = int(HIDDEN_SIZE * STUCK_PERCENTAGE)
    TEST_NAME = "Stuck-at-Zero-100%"
    N_RUNS    = 5

    for run_idx in range(N_RUNS):
        print(f"\n=== Stuck Run {run_idx+1}/{N_RUNS} ===")
        
        # Randomly pick which neurons to stick
        stuck_neuron_indices = random.sample(range(HIDDEN_SIZE), k=n_stuck)
        print("Stuck neuron indices:", stuck_neuron_indices)

        # Train
        model_stuck = train_alu_model_stuck_at_zero(
            device=device,
            epochs=2000,
            stuck_neuron_indices=stuck_neuron_indices
        )

        # Evaluate with stuck inference
        stuck_accuracy = evaluate_model_stuck_at_zero(
            model_stuck,
            device=device,
            stuck_neuron_indices=stuck_neuron_indices
        )

        # Log to CSV
        row = {
            "TestName": TEST_NAME,
            "RunIndex": run_idx + 1,
            "WhichNeuronsStuck": str(stuck_neuron_indices),
            "OriginalAccuracy": f"{normal_accuracy*100:.2f}",
            "StuckTrainAccuracy": f"{stuck_accuracy*100:.2f}"
        }
        append_result_to_csv(row)

    print(f"\nAll done! Results appended to {CSV_FILENAME}.")


if __name__ == "__main__":
    main()
