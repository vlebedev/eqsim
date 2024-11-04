import numpy as np
import cupy as cp
import numba
from numba import cuda, float32, int32
import matplotlib.pyplot as plt
from time import perf_counter

@cuda.jit
def compute_losses(magnitudes, epicenters, locations, values, types, soil_amp, losses):
    # Define shared arrays within a block
    shared_locs = cuda.shared.array(shape=(256,), dtype=float32)
    shared_vals = cuda.shared.array(shape=(256,), dtype=float32)
    shared_types = cuda.shared.array(shape=(256,), dtype=int32)
    shared_soil_amp = cuda.shared.array(shape=(256,), dtype=float32)

    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    dx = cuda.blockDim.x
    object_idx = tx

    while object_idx < locations.shape[0]:
        shared_locs[tx] = locations[object_idx]
        shared_vals[tx] = values[object_idx]
        shared_types[tx] = types[object_idx]
        shared_soil_amp[tx] = soil_amp[object_idx]
        object_idx += dx

    cuda.syncthreads()

    event_idx = cuda.grid(1)
    if event_idx < magnitudes.shape[0]:
        magnitude = magnitudes[event_idx]
        epicenter = epicenters[event_idx]
        total_loss = 0.0

        for i in range(locations.shape[0]):
            distance = abs(shared_locs[i] - epicenter)
            damage = max(0, (1 - 0.01 * distance) * (0.3 + 0.2 * shared_types[i] + 0.02 * magnitude + 0.1 * shared_soil_amp[i]))
            total_loss += damage * shared_vals[i]

        losses[event_idx] = total_loss

# Simulation parameters
numEvents = 100000
numInsuredObjects = 256*2  # Optimized for shared memory usage and full warp utilization
warmingUpIterations = 100

# Initialize data using CuPy
magnitudes = cp.random.uniform(low=2.0, high=7.0, size=numEvents).astype(np.float32)
epicenters = cp.random.uniform(low=0.0, high=100.0, size=numEvents).astype(np.float32)
locations = cp.linspace(1, 100, numInsuredObjects, dtype=np.float32)
values = cp.random.randint(300, 5000, size=numInsuredObjects).astype(np.float32)
types = cp.random.randint(1, 4, size=numInsuredObjects).astype(np.int32)
soil_amp = cp.random.randint(1, 4, size=numInsuredObjects).astype(np.float32)
losses = cp.zeros(numEvents, dtype=np.float32)

# Allocate device memory and copy data
magnitudes_gpu = cuda.to_device(magnitudes)
epicenters_gpu = cuda.to_device(epicenters)
locations_gpu = cuda.to_device(locations)
values_gpu = cuda.to_device(values)
types_gpu = cuda.to_device(types)
soil_amp_gpu = cuda.to_device(soil_amp)
losses_gpu = cuda.device_array_like(losses)

# Kernel execution configuration
threads_per_block = 256
blocks_per_grid = (numEvents + threads_per_block - 1) // threads_per_block

# Warm-up the GPU
for _ in range(warmingUpIterations):
    compute_losses[blocks_per_grid, threads_per_block](magnitudes_gpu, epicenters_gpu, locations_gpu, values_gpu, types_gpu, soil_amp_gpu, losses_gpu)
    # Zero out device array
    losses_gpu = cuda.device_array(numEvents, dtype=np.float32)

# Run simulation
start_time = perf_counter()
compute_losses[blocks_per_grid, threads_per_block](magnitudes_gpu, epicenters_gpu, locations_gpu, values_gpu, types_gpu, soil_amp_gpu, losses_gpu)
cuda.synchronize()
end_time = perf_counter()

losses = losses_gpu.copy_to_host()
# Sort the losses in descending order for further analysis
sorted_losses = np.sort(losses)[::-1]


print(f"GPU Computation Time: {(end_time - start_time)*1000:.6f} ms")
print(f"Maximum loss: {np.max(losses):.2f}")
print(f"Average loss: {np.mean(losses):.2f}")


