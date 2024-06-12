## Naive Earthquake Damage Calculation Model
##
## PyTorch implementation
## 
## Model assumptions:
## - All insured properties (assets) located on the EQ fault line and spead evenly
## - Each property has location, value, construction type and soil amplification parameters
## - Each EQ event has epicenter location and magnitude
##
## Damage function is defined as follows:
## damage = max(0, (1.0 - 0.01 * distance_from_epicenter) * (0.3 + 0.2 * construction_type + 0.02 * magnitude + 0.1 * amplification))
##
## Make sure it is opened in devcontainer, then run it with the following:
## # python3 simpl_eq_pytorch.py <numEvents> <numProperties> <warmingUpIterations>

import sys
import torch
import time

def compute_losses(magnitudes, epicenters, locations, values, types, soil_amp):
    # Expand dimensions to allow vectorized computation across all events and locations
    magnitudes = magnitudes[:, None]  # Shape: [numEvents, 1]
    epicenters = epicenters[:, None]  # Shape: [numEvents, 1]
    distances = torch.abs(locations - epicenters)  # Broadcast subtraction over the second dimension

    # Calculate damage for each location for each event
    damages = torch.clamp(1 - 0.01 * distances, min=0) * (0.3 + 0.2 * types + 0.02 * magnitudes + 0.1 * soil_amp)
    total_losses = torch.sum(damages * values, dim=1)  # Sum over locations for each event
    return total_losses

# Ensure correct number of command-line arguments
if len(sys.argv) != 4:
    print("Usage: python3 simpl_eq_pytorch.py <numEvents> <numProperties> <warmingUpIterations>")
    sys.exit(1)

numEvents = int(sys.argv[1])
numInsuredObjects = int(sys.argv[2])
warmingUpIterations = int(sys.argv[3])

print(f"Simulating earthquake damages for {numEvents} events and {numInsuredObjects} properties with {warmingUpIterations} warming-up iterations.")

# Simulation parameters
faultLineLength = 100

# Initialize data for both CPU and GPU
cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to initialize data
def init_data(device):
    magnitudes = torch.rand(numEvents, device=device) * 5 + 2  # Uniformly distributed between 2 and 7
    epicenters = torch.rand(numEvents, device=device) * faultLineLength
    locations = torch.linspace(1, faultLineLength, numInsuredObjects, device=device)
    values = torch.randint(300, 5000, (numInsuredObjects,), dtype=torch.float32, device=device)
    types = torch.randint(1, 4, (numInsuredObjects,), dtype=torch.float32, device=device)
    soil_amp = torch.randint(1, 4, (numInsuredObjects,), dtype=torch.float32, device=device)
    return magnitudes, epicenters, locations, values, types, soil_amp

# Function to run simulation
def run_simulation(device_name, device):
    print(f"\nRunning on {device_name}")
    magnitudes, epicenters, locations, values, types, soil_amp = init_data(device)
    
    # Warm-up the device
    for _ in range(warmingUpIterations):
        _ = compute_losses(magnitudes, epicenters, locations, values, types, soil_amp)

    # Start the timer
    start_time = time.time()

    # Compute losses
    losses = compute_losses(magnitudes, epicenters, locations, values, types, soil_amp)

    # Stop the timer
    end_time = time.time()

    # Compute statistics
    avg_loss = torch.mean(losses)
    sorted_losses = torch.sort(losses, descending=True)[0]

    # Print results
    print(f"Computation Time on {device_name}: {(end_time - start_time)*1000:.6f} ms")
    print(f"Maximum loss: {sorted_losses[0].item():.2f}")
    print(f"Average loss: {avg_loss.item():.2f}")

# Run simulation on CPU
run_simulation("CPU", cpu_device)

# Run simulation on CUDA, if available
if cuda_device != cpu_device:
    run_simulation(torch.cuda.get_device_name(cuda_device.index), cuda_device)
