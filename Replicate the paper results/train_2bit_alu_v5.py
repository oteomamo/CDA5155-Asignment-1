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
    """
    2-bit ALU logic:
    00 => AND, 01 => OR, 10 => ADD(mod4), 11 => XOR
    """
    if opcode == 0:      # 00 => AND
        return a & b
    elif opcode == 1:    # 01 => OR
        return a | b
    elif opcode == 2:    # 10 => ADD (mod 4)
        return (a + b) % 4
    else:                # 11 => XOR
        return a ^ b

def generate_dataset():
    """
    Creates the 64-sample dataset for the 2-bit ALU:
    Inputs: [A0,A1,B0,B1,Op0,Op1]
    Outputs: [Out0,Out1] (Out0=LSB, Out1=MSB)
    """
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
    """
    Train the ALU model normally (no fault injection).
    """
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
    """
    Evaluate with no errors/faults.
    Returns accuracy in [0..1].
    """
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
# 4) Stuck-at-Zero TRAINING (Mask-based) - (Existing code)
##############################################################################
def train_alu_model_stuck_at_zero(device='cpu', epochs=2000, stuck_neuron_indices=None):
    """
    Train the model while certain hidden neurons are "stuck at zero."
    """
    if stuck_neuron_indices is None:
        stuck_neuron_indices = []

    x_train, y_train = generate_dataset()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ALUNet(hidden_size=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    neuron_mask = torch.ones(model.fc1.out_features, device=device)
    for idx in stuck_neuron_indices:
        neuron_mask[idx] = 0.0

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        hidden = model.fc1(x_train)
        hidden = torch.relu(hidden)

        batch_size = hidden.size(0)
        mask = neuron_mask.unsqueeze(0).expand(batch_size, -1)
        hidden = hidden * mask

        logits = model.fc2(hidden)

        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"[Stuck-Train-Zero] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model

def evaluate_model_stuck_at_zero(model, device='cpu', stuck_neuron_indices=None):
    """
    Evaluate with hidden neurons forced to zero in the hidden layer.
    Returns accuracy in [0..1].
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
# 5) New: Stuck-at-High (Neuron Output Saturation)
##############################################################################
def train_alu_model_stuck_at_high(device='cpu',
                                  epochs=2000,
                                  stuck_neuron_indices=None,
                                  saturation_value=1.0):
    """
    Train the model while certain hidden neurons are forced to a constant
    'saturation_value' (e.g. 1.0).
    This is analogous to 'Stuck-at-Zero', but now the chosen neurons
    always output a fixed high value after ReLU.
    """
    if stuck_neuron_indices is None:
        stuck_neuron_indices = []

    x_train, y_train = generate_dataset()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ALUNet(hidden_size=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # We create two masks:
    # - mask_normal = 1 for normal neurons, 0 for stuck
    # - mask_stuck  = 0 for normal neurons, 1 for stuck
    # Then hidden_out = hidden * mask_normal + saturation_value * mask_stuck
    mask_normal = torch.ones(model.fc1.out_features, device=device)
    mask_stuck  = torch.zeros(model.fc1.out_features, device=device)
    for idx in stuck_neuron_indices:
        mask_normal[idx] = 0.0
        mask_stuck[idx]  = 1.0

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        hidden = model.fc1(x_train)
        hidden = torch.relu(hidden)

        batch_size = hidden.size(0)
        normal_mask = mask_normal.unsqueeze(0).expand(batch_size, -1)
        stuck_mask  = mask_stuck.unsqueeze(0).expand(batch_size, -1)

        # Combine
        hidden = hidden * normal_mask + saturation_value * stuck_mask

        logits = model.fc2(hidden)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"[Stuck-Train-High] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model

def evaluate_model_stuck_at_high(model,
                                 device='cpu',
                                 stuck_neuron_indices=None,
                                 saturation_value=1.0):
    """
    Evaluate the model with certain neurons forced to a constant
    'saturation_value' in the hidden layer.
    """
    if stuck_neuron_indices is None:
        stuck_neuron_indices = []

    x_test, y_test = generate_dataset()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    mask_normal = torch.ones(model.fc1.out_features, device=device)
    mask_stuck  = torch.zeros(model.fc1.out_features, device=device)
    for idx in stuck_neuron_indices:
        mask_normal[idx] = 0.0
        mask_stuck[idx]  = 1.0

    with torch.no_grad():
        hidden = model.fc1(x_test)
        hidden = torch.relu(hidden)

        batch_size = hidden.size(0)
        normal_mask = mask_normal.unsqueeze(0).expand(batch_size, -1)
        stuck_mask  = mask_stuck.unsqueeze(0).expand(batch_size, -1)

        hidden = hidden * normal_mask + saturation_value * stuck_mask

        logits = model.fc2(hidden)
        preds = torch.sigmoid(logits)
        predicted_bits = (preds > 0.5).int()
        correct_bits   = (predicted_bits == y_test.int())
        accuracy = correct_bits.all(dim=1).float().mean().item()

    print(f"Stuck-at-High (neurons={stuck_neuron_indices}, val={saturation_value}), Accuracy: {accuracy*100:.2f}%")
    return accuracy



##############################################################################
# 6) Training with Gaussian Noise in Hidden Activations
##############################################################################
def train_alu_model_with_gaussian_noise_in_hidden(device='cpu', 
                                                  epochs=2000, 
                                                  noise_std=0.01):
    """
    Train the model while adding small Gaussian noise to the hidden activations
    in each forward pass:
        hidden = ReLU(fc1(x)) + N(0, noise_std^2)
    """
    x_train, y_train = generate_dataset()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ALUNet(hidden_size=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1) Forward pass up to hidden
        hidden = model.fc1(x_train)
        hidden = torch.relu(hidden)

        # 2) Add noise: shape [64, 8] for hidden
        # We'll do in a functional (non-in-place) way:
        noise = torch.normal(mean=0.0,
                             std=noise_std,
                             size=hidden.shape,
                             device=device)
        hidden_noisy = hidden + noise

        # 3) Output layer
        logits = model.fc2(hidden_noisy)

        # 4) Backprop
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"[Train w/ Hidden Noise std={noise_std}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model

def evaluate_model_with_gaussian_noise_in_hidden(model,
                                                 device='cpu',
                                                 noise_std=0.01):
    """
    Evaluate the model but add noise to hidden activations:
        hidden = ReLU(fc1(x)) + noise
    This simulates inference-time noise.
    """
    x_test, y_test = generate_dataset()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        hidden = model.fc1(x_test)
        hidden = torch.relu(hidden)

        noise = torch.normal(mean=0.0, std=noise_std, size=hidden.shape, device=device)
        hidden_noisy = hidden + noise

        logits = model.fc2(hidden_noisy)
        preds = torch.sigmoid(logits)
        predicted_bits = (preds > 0.5).int()
        correct_bits   = (predicted_bits == y_test.int())
        accuracy = correct_bits.all(dim=1).float().mean().item()

    print(f"Evaluate w/ Hidden Noise (std={noise_std}), Accuracy: {accuracy*100:.2f}%")
    return accuracy

##############################################################################
# 7) Helper: Apply Combinational Errors
##############################################################################
def apply_combinational_errors(hidden, 
                               p_stuck_zero=0.0,
                               p_stuck_high=0.0,
                               p_noise=0.0,
                               high_val=1.0,
                               noise_std=0.01):
    """
    We process each neuron's hidden activation independently.
    For each neuron, we do 3 independent checks:
      1) With probability p_stuck_zero, we set it to 0.
      2) With probability p_stuck_high, we set it to high_val.
      3) With probability p_noise, we add Gaussian noise N(0, noise_std^2).
    Because these checks are independent, the neuron could have multiple
    errors in one pass (e.g., stuck at high, then get noise added).
    The order here is:
       stuck-zero => stuck-high => add-noise
    So if a neuron is stuck-zero, it might get overwritten by stuck-high,
    and then also get noise. Adjust the order if you prefer a different logic.
    """
    # hidden shape: [batch_size, hidden_size]
    device = hidden.device
    batch_size, hidden_size = hidden.shape

    # We'll do everything in a functional way (no in-place ops).
    out = hidden.clone()

    # 1) Stuck-at-zero
    #   We'll create a mask of shape [batch_size, hidden_size]
    #   where each entry is 1 if it was NOT stuck, or 0 if stuck.
    if p_stuck_zero > 0.0:
        rand_zero = torch.rand_like(out)  # uniform in [0,1], same shape
        zero_mask = (rand_zero > p_stuck_zero).float()  # 1 if no-stuck, 0 if stuck
        out = out * zero_mask  # those that stuck become zero

    # 2) Stuck-at-high
    if p_stuck_high > 0.0:
        rand_high = torch.rand_like(out)
        # We'll add (high_val - out) to those that are stuck
        # i.e. out -> high_val
        stuck_mask = (rand_high <= p_stuck_high).float()
        out = out * (1 - stuck_mask) + high_val * stuck_mask

    # 3) Gaussian noise
    if p_noise > 0.0:
        rand_noise = torch.rand_like(out)
        # For each element, if rand_noise[i,j] <= p_noise => we add noise
        noise_mask = (rand_noise <= p_noise).float()
        # Create a noise tensor for all elements
        noise_tensor = torch.normal(mean=0.0, std=noise_std, size=out.shape, device=device)
        # Only add noise where noise_mask==1
        out = out + noise_mask * noise_tensor

    return out


##############################################################################
# 8) Train with Combinational Errors
##############################################################################
def train_alu_model_combinational_errors(device='cpu',
                                         epochs=2000,
                                         p_stuck_zero=0.1,
                                         p_stuck_high=0.1,
                                         p_noise=0.1,
                                         high_val=1.0,
                                         noise_std=0.01):
    """
    In each forward pass, we do:
      hidden = ReLU(fc1(x))
      hidden_err = apply_combinational_errors(hidden, p_stuck_zero, p_stuck_high, p_noise, ...)
      logits = fc2(hidden_err)
    Then we backprop as usual.

    Because errors are random each time, the network sees a variety of
    fault combinations on each epoch and tries to learn to cope with them.
    """
    x_train, y_train = generate_dataset()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ALUNet(hidden_size=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()

        hidden = model.fc1(x_train)
        hidden = torch.relu(hidden)

        hidden_err = apply_combinational_errors(hidden,
                                                p_stuck_zero=p_stuck_zero,
                                                p_stuck_high=p_stuck_high,
                                                p_noise=p_noise,
                                                high_val=high_val,
                                                noise_std=noise_std)

        logits = model.fc2(hidden_err)

        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"[ComboErr] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model


##############################################################################
# 9) Evaluate with Combinational Errors
##############################################################################
def evaluate_model_combinational_errors(model,
                                        device='cpu',
                                        p_stuck_zero=0.1,
                                        p_stuck_high=0.1,
                                        p_noise=0.1,
                                        high_val=1.0,
                                        noise_std=0.01):
    """
    Evaluate the model with the same combinational error injection approach
    at inference time. Each neuron's hidden activation could be zeroed,
    saturated, or noised, independently.

    If you want to see how the model does with "pure" inference (no errors),
    just call evaluate_model(model).
    """
    x_test, y_test = generate_dataset()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        hidden = model.fc1(x_test)
        hidden = torch.relu(hidden)

        hidden_err = apply_combinational_errors(hidden,
                                                p_stuck_zero=p_stuck_zero,
                                                p_stuck_high=p_stuck_high,
                                                p_noise=p_noise,
                                                high_val=high_val,
                                                noise_std=noise_std)

        logits = model.fc2(hidden_err)
        preds = torch.sigmoid(logits)
        predicted_bits = (preds > 0.5).int()
        correct_bits   = (predicted_bits == y_test.int())
        accuracy = correct_bits.all(dim=1).float().mean().item()

    print(f"Combinational Errors (pZ={p_stuck_zero}, pH={p_stuck_high}, pN={p_noise}), Acc={accuracy*100:.2f}%")
    return accuracy


##############################################################################
# 10) Logging to CSV
##############################################################################
CSV_FILENAME = "experiments_results_v5.csv"

def append_result_to_csv(row_dict):
    file_exists = os.path.isfile(CSV_FILENAME)
    fieldnames = [
        "TestName",
        "RunIndex",
        "WhichNeuronsStuck",  # We'll leave it as a placeholder
        "OriginalAccuracy",
        "StuckTrainAccuracy"  # We'll store final accuracy here
    ]
    with open(CSV_FILENAME, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)



##############################################################################
# 11) MAIN
##############################################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #------------------------------------------------------------------
    # 1) Train normal model once (baseline)
    #------------------------------------------------------------------
    model_normal = train_alu_model(device=device, epochs=2000)
    normal_accuracy = evaluate_model(model_normal, device=device)

    #------------------------------------------------------------------
    # NOTE: We'll comment out the stuck-at-zero portion just to show we keep it around
    """
    # 2) Stuck-at-Zero TRAINING: 25% random neurons x 5 runs
    STUCK_PERCENTAGE = 0.25
    HIDDEN_SIZE       = 8
    n_stuck = int(HIDDEN_SIZE * STUCK_PERCENTAGE)
    TEST_NAME = "Stuck-at-Zero-25%"
    N_RUNS    = 5

    for run_idx in range(N_RUNS):
        stuck_neuron_indices = random.sample(range(HIDDEN_SIZE), k=n_stuck)
        model_stuck = train_alu_model_stuck_at_zero(
            device=device,
            epochs=2000,
            stuck_neuron_indices=stuck_neuron_indices
        )
        stuck_accuracy = evaluate_model_stuck_at_zero(
            model_stuck,
            device=device,
            stuck_neuron_indices=stuck_neuron_indices
        )
        row = {
            "TestName": TEST_NAME,
            "RunIndex": run_idx + 1,
            "WhichNeuronsStuck": str(stuck_neuron_indices),
            "OriginalAccuracy": f"{normal_accuracy*100:.2f}",
            "StuckTrainAccuracy": f"{stuck_accuracy*100:.2f}"
        }
        append_result_to_csv(row)
    """

    #------------------------------------------------------------------
    # 2) STUCK-AT-HIGH (RANDOM) - e.g. 25% saturation, 5 runs
    #------------------------------------------------------------------
    """
    SAT_HIGH_PERCENTAGE = 1
    HIDDEN_SIZE = 8
    n_stuck_high = int(HIDDEN_SIZE * SAT_HIGH_PERCENTAGE)
    TEST_NAME_HIGH_RAND = "Stuck-at-High-Random-100%"
    N_RUNS = 5
    SATURATION_VALUE = 1.0  # or larger if you want

    for run_idx in range(N_RUNS):
        print(f"\n[Stuck-at-High Random] Run {run_idx+1}/{N_RUNS}")
        stuck_neurons = random.sample(range(HIDDEN_SIZE), k=n_stuck_high)

        model_stuck_high = train_alu_model_stuck_at_high(
            device=device,
            epochs=2000,
            stuck_neuron_indices=stuck_neurons,
            saturation_value=SATURATION_VALUE
        )
        stuck_accuracy = evaluate_model_stuck_at_high(
            model_stuck_high,
            device=device,
            stuck_neuron_indices=stuck_neurons,
            saturation_value=SATURATION_VALUE
        )

        row = {
            "TestName": TEST_NAME_HIGH_RAND,
            "RunIndex": run_idx + 1,
            "WhichNeuronsStuck": str(stuck_neurons),
            "OriginalAccuracy": f"{normal_accuracy*100:.2f}",
            "StuckTrainAccuracy": f"{stuck_accuracy*100:.2f}"
        }
        append_result_to_csv(row)

    """
    #------------------------------------------------------------------
    # 3) STUCK-AT-HIGH (SPECIFIC) - user-specified neurons
    #    We do 5 runs as well, but re-using the same subset each time,
    #    or you can vary it. Here we'll just show a single example set:
    #------------------------------------------------------------------
    """
    SPECIFIC_NEURONS = [2, 5]  # user-chosen example
    TEST_NAME_HIGH_SPECIFIC = "Stuck-at-High-Specific-[2,5]"
    N_RUNS_SPECIFIC = 5

    for run_idx in range(N_RUNS_SPECIFIC):
        print(f"\n[Stuck-at-High Specific] Run {run_idx+1}/{N_RUNS_SPECIFIC}")
        # We always saturate neurons 2 and 5 in each run
        model_stuck_high_spec = train_alu_model_stuck_at_high(
            device=device,
            epochs=2000,
            stuck_neuron_indices=SPECIFIC_NEURONS,
            saturation_value=SATURATION_VALUE
        )
        stuck_accuracy_spec = evaluate_model_stuck_at_high(
            model_stuck_high_spec,
            device=device,
            stuck_neuron_indices=SPECIFIC_NEURONS,
            saturation_value=SATURATION_VALUE
        )

        row = {
            "TestName": TEST_NAME_HIGH_SPECIFIC,
            "RunIndex": run_idx + 1,
            "WhichNeuronsStuck": str(SPECIFIC_NEURONS),
            "OriginalAccuracy": f"{normal_accuracy*100:.2f}",
            "StuckTrainAccuracy": f"{stuck_accuracy_spec*100:.2f}"
        }
        append_result_to_csv(row)
    """
    
    
    #------------------------------------------------------------------
    # 4) GAUSSIAN NOISE EXAMPLES
    #------------------------------------------------------------------
    # Example A: Train with noise, Evaluate with noise
    # We do 5 runs, each run can keep the same noise_std or vary it.
    # Different noise standard deviations (0.01, 0.05, 0.1, 0.2)
    """
    NOISE_STD = 0.2
    TEST_NAME = f"TrainNoise-Hidden-{NOISE_STD}"
    N_RUNS = 5

    for run_idx in range(N_RUNS):
        print(f"\n[Train w/ Hidden Noise, run {run_idx+1}/{N_RUNS}, std={NOISE_STD}]")
        model_noise = train_alu_model_with_gaussian_noise_in_hidden(device=device,
                                                                    epochs=2000,
                                                                    noise_std=NOISE_STD)
        final_acc = evaluate_model_with_gaussian_noise_in_hidden(model_noise,
                                                                 device=device,
                                                                 noise_std=NOISE_STD)

        # Log to CSV
        row = {
            "TestName": TEST_NAME,
            "RunIndex": run_idx + 1,
            "WhichNeuronsStuck": "None",
            "OriginalAccuracy": f"{normal_accuracy*100:.2f}",
            "StuckTrainAccuracy": f"{final_acc*100:.2f}"
        }
        append_result_to_csv(row)

    # Example B: Train normally, Evaluate with noise
    # Compare how a normal model handles inference-time noise
    TEST_NAME_EVAL_NOISE = f"EvalNoise-Hidden-{NOISE_STD}"
    for run_idx in range(N_RUNS):
        print(f"\n[Eval w/ Hidden Noise, run {run_idx+1}/{N_RUNS}, std={NOISE_STD}]")
        final_acc = evaluate_model_with_gaussian_noise_in_hidden(model_normal,
                                                                 device=device,
                                                                 noise_std=NOISE_STD)

        row = {
            "TestName": TEST_NAME_EVAL_NOISE,
            "RunIndex": run_idx + 1,
            "WhichNeuronsStuck": "None",
            "OriginalAccuracy": f"{normal_accuracy*100:.2f}",
            "StuckTrainAccuracy": f"{final_acc*100:.2f}"
        }
        append_result_to_csv(row)
        
    """


    #------------------------------------------------------------------
    # 5) Combinational Errors
    #    Example: p_stuck_zero=0.1, p_stuck_high=0.2, p_noise=0.3
    #    We'll run 5 times with the same probabilities, but random draws each run.
    #------------------------------------------------------------------
    pZ = 0.2  # Probability that a neuron is forced to zero
    pH = 0.3  # Probability that a neuron is forced to 'high_val'
    pN = 0.4  # Probability that a neuron gets noise
    NOISE_STD = 0.1
    HIGH_VAL = 1.0

    TEST_NAME = f"ComboErr_Z{pZ}_H{pH}_N{pN}"
    N_RUNS = 5

    for run_idx in range(N_RUNS):
        print(f"\n=== Combination Errors Run {run_idx+1}/{N_RUNS} ===")
        # Train the model with these probabilities
        model_combo = train_alu_model_combinational_errors(device=device,
                                                           epochs=2000,
                                                           p_stuck_zero=pZ,
                                                           p_stuck_high=pH,
                                                           p_noise=pN,
                                                           high_val=HIGH_VAL,
                                                           noise_std=NOISE_STD)
        # Evaluate with the same errors
        combo_acc = evaluate_model_combinational_errors(model_combo,
                                                        device=device,
                                                        p_stuck_zero=pZ,
                                                        p_stuck_high=pH,
                                                        p_noise=pN,
                                                        high_val=HIGH_VAL,
                                                        noise_std=NOISE_STD)
        # Log
        row = {
            "TestName": TEST_NAME,
            "RunIndex": run_idx + 1,
            "WhichNeuronsStuck": "RandomEachForward",  # It's random each pass
            "OriginalAccuracy": f"{normal_accuracy*100:.2f}",
            "StuckTrainAccuracy": f"{combo_acc*100:.2f}"
        }
        append_result_to_csv(row)

    print(f"\nAll done! Results appended to {CSV_FILENAME}.")


if __name__ == "__main__":
    main()