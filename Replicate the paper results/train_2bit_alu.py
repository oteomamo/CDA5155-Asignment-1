import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------
# 1) ALU logic function
# ---------------------------------------------
def alu_operation(a, b, opcode):
    """
    a, b are integers 0..3 (2-bit),
    opcode is also 0..3 (2-bit),
    returns an integer 0..3 representing 2-bit output,
    ignoring carry for ADD (mod 4).

    Mapping:
    00 (opcode=0) -> AND
    01 (opcode=1) -> OR
    10 (opcode=2) -> (A + B) % 4
    11 (opcode=3) -> XOR
    """
    if opcode == 0:      # 00 => AND
        return a & b
    elif opcode == 1:    # 01 => OR
        return a | b
    elif opcode == 2:    # 10 => ADD (mod 4)
        return (a + b) % 4
    else:                # 11 => XOR
        return a ^ b

# ---------------------------------------------
# 2) Generate the full dataset (64 samples)
# ---------------------------------------------
def generate_dataset():
    """
    Each sample has 6 inputs (A0,A1,B0,B1,Op0,Op1) and 2 outputs (Out0,Out1)
    Out0 is the LSB, Out1 is the MSB.
    """
    inputs = []
    outputs = []

    for A in range(4):       # 0..3 for 2-bit A
        for B in range(4):   # 0..3 for 2-bit B
            for op in range(4):  # 0..3 for opcode
                # Convert A, B, opcode to 6 binary inputs
                A_bin = format(A, '02b')  # e.g. '00', '01', '10', '11'
                B_bin = format(B, '02b')
                op_bin= format(op, '02b')

                # We'll parse them as bits in the order [A0, A1, B0, B1, Op0, Op1].
                A0, A1 = int(A_bin[1]), int(A_bin[0])
                B0, B1 = int(B_bin[1]), int(B_bin[0])
                Op0, Op1= int(op_bin[1]), int(op_bin[0])

                inp = [A0, A1, B0, B1, Op0, Op1]

                # Compute the 2-bit result => an integer 0..3
                result = alu_operation(A, B, op)
                # Convert that to 2 bits: Out0 (LSB) and Out1 (MSB)
                res_bin = format(result, '02b')
                Out0, Out1 = int(res_bin[1]), int(res_bin[0])

                inputs.append(inp)
                outputs.append([Out0, Out1])

    # Convert to PyTorch tensors
    inputs_tensor  = torch.tensor(inputs,  dtype=torch.float32)  # shape [64, 6]
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)  # shape [64, 2]

    return inputs_tensor, outputs_tensor

# ---------------------------------------------
# 3) Define the MLP
# ---------------------------------------------
class ALUNet(nn.Module):
    def __init__(self, hidden_size=8):
        super(ALUNet, self).__init__()
        # 6 inputs -> hidden_size -> 2 outputs
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        # We'll use BCEWithLogitsLoss, so no final Sigmoid here.

    def forward(self, x):
        # Hidden layer with ReLU
        x = self.fc1(x)
        x = torch.relu(x)
        # Output layer (raw logits for each bit)
        x = self.fc2(x)
        return x

# ---------------------------------------------
# 4) Normal Training
# ---------------------------------------------
def train_alu_model(device='cpu', epochs=2000):
    """
    Normal training: no stuck-at-zero. The entire layer is functional.
    """
    x_train, y_train = generate_dataset()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ALUNet(hidden_size=8).to(device)
    criterion = nn.BCEWithLogitsLoss()  # For 2 binary outputs
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)             # shape [64, 2]
        loss   = criterion(y_pred, y_train) # BCE with logits
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"[Normal Train] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model

# ---------------------------------------------
# 5) Normal Evaluation (no errors)
# ---------------------------------------------
def evaluate_model(model, device='cpu'):
    """
    Evaluate the model in a normal scenario (no error injection).
    """
    x_test, y_test = generate_dataset()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        logits = model(x_test)          # shape [64,2]
        preds = torch.sigmoid(logits)   # bit probabilities
        predicted_bits = (preds > 0.5).int()
        correct_bits   = (predicted_bits == y_test.int())

        # A sample is correct if both bits match the ground truth
        correct_per_sample = correct_bits.all(dim=1)
        accuracy = correct_per_sample.float().mean().item()

    print(f"Accuracy on 64 patterns (no error): {accuracy*100:.2f}%")

# ---------------------------------------------
# 6) Stuck-at-Zero TRAINING
# ---------------------------------------------
def train_alu_model_stuck_at_zero(device='cpu', epochs=2000, stuck_neuron_indices=None):
    """
    Train the model while certain hidden neurons are "stuck at zero."
    That means for each forward pass *during training*, we zero out
    specific neuron outputs in the hidden layer.

    stuck_neuron_indices: list of neuron IDs to clamp at zero after ReLU.
    Example: [0,1] means the first two neurons in the hidden layer are always 0.
    """
    if stuck_neuron_indices is None:
        stuck_neuron_indices = []

    x_train, y_train = generate_dataset()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ALUNet(hidden_size=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Instead of calling model(x_train) directly, we do:
        # 1) fc1, 2) ReLU, 3) zero out the stuck neurons, 4) fc2
        hidden = model.fc1(x_train)
        hidden = torch.relu(hidden)

        # Force chosen neurons to 0
        for idx in stuck_neuron_indices:
            hidden[:, idx] = 0.0

        logits = model.fc2(hidden)

        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"[Stuck-Train] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model

# ---------------------------------------------
# 7) Evaluate with Stuck-at-Zero (Inference)
# ---------------------------------------------
def evaluate_model_stuck_at_zero(model, device='cpu', stuck_neuron_indices=None):
    """
    Evaluate the model with certain neurons forced to zero in the hidden layer
    at inference time.
    """
    if stuck_neuron_indices is None:
        stuck_neuron_indices = []

    x_test, y_test = generate_dataset()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        hidden = model.fc1(x_test)
        hidden = torch.relu(hidden)

        for idx in stuck_neuron_indices:
            hidden[:, idx] = 0.0

        logits = model.fc2(hidden)
        preds = torch.sigmoid(logits)
        predicted_bits = (preds > 0.5).int()
        correct_bits   = (predicted_bits == y_test.int())
        correct_per_sample = correct_bits.all(dim=1)
        accuracy = correct_per_sample.float().mean().item()

    print(f"Stuck-at-Zero (neurons={stuck_neuron_indices}), Accuracy: {accuracy*100:.2f}%")

# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ############################################################################
    # 1) NORMAL TRAINING & EVALUATION
    ############################################################################
    model_normal = train_alu_model(device=device, epochs=2000)
    evaluate_model(model_normal, device=device)

    # Evaluate the same model with stuck neurons at inference (not trained that way)
    evaluate_model_stuck_at_zero(model_normal, device=device, stuck_neuron_indices=[0,1])

    ############################################################################
    # 2) STUCK-AT-ZERO TRAINING
    ############################################################################
    # Now we train a separate model but keep neurons stuck at zero DURING training
    stuck_neurons = [0,1]  # pick whichever neurons you want to disable
    print("\n=== Training WITH stuck-at-zero (neurons: {}) ===".format(stuck_neurons))
    model_stuck = train_alu_model_stuck_at_zero(device=device, epochs=2000,
                                                stuck_neuron_indices=stuck_neurons)
    # Evaluate with "no" error injection to see how well it does if those neurons
    # were effectively never used
    print("Evaluating the stuck-trained model with normal (no-error) inference:")
    evaluate_model(model_stuck, device=device)

    # Evaluate with the same stuck neurons at inference (should match training condition)
    print("Evaluating the stuck-trained model with matching stuck-at-zero inference:")
    evaluate_model_stuck_at_zero(model_stuck, device=device, stuck_neuron_indices=stuck_neurons)

if __name__ == "__main__":
    main()
