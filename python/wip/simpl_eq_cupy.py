import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt
from time import perf_counter

# Define the CUDA kernel for computing losses
@cuda.jit
def compute_losses(magnitudes, epicenters, locations, values, types, soil_amp, losses):
    idx = cuda.grid(1)
    if idx < losses.size:
        total_loss = 0.0
        magnitude = magnitudes[idx]
        epicenter = epicenters[idx]
        for j in range(locations.size):
            distance = abs(locations[j] - epicenter)
            damage = max(0, (1 - 0.01 * distance) * (0.3 + 0.2 * types[j] + 0.02 * magnitude + 0.1 * soil_amp[j]))
            total_loss += damage * values[j]
        losses[idx] = total_loss

# Simulation parameters
numEvents = 100000
numInsuredObjects = 1000
faultLineLength = 100

# Initialize data using CuPy
magnitudes = cp.random.uniform(low=2.0, high=7.0, size=numEvents)
epicenters = cp.random.uniform(low=0.0, high=faultLineLength, size=numEvents)
locations = cp.linspace(1, faultLineLength, numInsuredObjects)
values = cp.random.randint(300, 5000, size=numInsuredObjects)
types = cp.random.randint(1, 4, size=numInsuredObjects)
soil_amp = cp.random.randint(1, 4, size=numInsuredObjects)

# Array for storing the results
losses = cp.zeros(numEvents, dtype=cp.float32)

# Compute grid dimensions
threads_per_block = 256
blocks_per_grid = (numEvents + (threads_per_block - 1)) // threads_per_block

# Timing start
start_time = perf_counter()

# Launch the kernel
compute_losses[blocks_per_grid, threads_per_block](magnitudes, epicenters, locations, values, types, soil_amp, losses)

# Sync device to ensure completion
cp.cuda.runtime.deviceSynchronize()

# Timing end
end_time = perf_counter()
elapsed_time = end_time - start_time
# print(f"Using GPU: {cp.cuda.Device(0).name}")
print(f"GPU Computation Time: {elapsed_time*1000:.6f} ms")

# Sort the losses for further analysis
sorted_losses = cp.sort(losses)

# Transfer data from device to host
sorted_losses_cpu = cp.asnumpy(sorted_losses)

# Compute maximum and average losses
max_loss = cp.max(losses).get()
avg_loss = cp.mean(losses).get()
print(f"Maximum loss: {max_loss:.2f}")
print(f"Average loss: {avg_loss:.2f}")


