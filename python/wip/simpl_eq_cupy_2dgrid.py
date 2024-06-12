import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt
from time import perf_counter

# Define the CUDA kernel for computing damages for each event-object pair
@cuda.jit
def compute_damages(magnitudes, epicenters, locations, values, types, soil_amp, losses):
    event_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    object_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if event_idx < magnitudes.size and object_idx < locations.size:
        distance = abs(locations[object_idx] - epicenters[event_idx])
        damage = max(0, (1 - 0.01 * distance) * (0.3 + 0.2 * types[object_idx] + 0.02 * magnitudes[event_idx] + 0.1 * soil_amp[object_idx]))
        cuda.atomic.add(losses, event_idx, damage * values[object_idx])

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
threads_per_block = (8, 8)  # 16x16 threads per block
blocks_per_grid_x = (numEvents + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (numInsuredObjects + threads_per_block[1] - 1) // threads_per_block[1]

# Timing start
start_time = perf_counter()

# Launch the kernel
compute_damages[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](magnitudes, epicenters, locations, values, types, soil_amp, losses)

# Sync device to ensure completion
cp.cuda.runtime.deviceSynchronize()

# Timing end
end_time = perf_counter()
elapsed_time = end_time - start_time
print(f"GPU Computation Time: {elapsed_time*1000:.6f} ms")

# Compute maximum and average losses
max_loss = cp.max(losses).get()
avg_loss = cp.mean(losses).get()
print(f"Maximum loss: {max_loss:.2f}")
print(f"Average loss: {avg_loss:.2f}")

# Sort the losses for analysis
sorted_losses = cp.sort(losses)

# Transfer data from device to host for further analysis
sorted_losses_cpu = cp.asnumpy(sorted_losses)
