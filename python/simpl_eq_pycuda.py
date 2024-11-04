## Naive Earthquake Damage Calculation Model
##
## PyCUDA implementation
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
## # python3 simpl_eq_pycuda.py <numEvents> <numProperties> <warmingUpIterations>

import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from time import perf_counter

# Define the CUDA kernel in C-like syntax
kernel_code = """
__global__ void compute_damages(float *magnitudes, float *epicenters, float *locations, float *values, int *types, float *soil_amp, float *losses, int numEvents, int numObjects)
{
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int object_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (event_idx < numEvents && object_idx < numObjects) {
        float distance = fabs(locations[object_idx] - epicenters[event_idx]);
        float damage = max(0.0, (1.0 - 0.01 * distance) * (0.3 + 0.2 * types[object_idx] + 0.02 * magnitudes[event_idx] + 0.1 * soil_amp[object_idx]));
        atomicAdd(&losses[event_idx], damage * values[object_idx]);
    }
}
"""

# Ensure correct number of command line arguments
if len(sys.argv) != 4:
    print("Usage: python3 simpl_eq_pycuda.py <numEvents> <numInsuredObjects> <warmingUpIterations>")
    sys.exit(1)

numEvents = int(sys.argv[1])
numInsuredObjects = int(sys.argv[2])
warmingUpIterations = int(sys.argv[3])

print(f"Simulating earthquake damages for {numEvents} events and {numInsuredObjects} properties with {warmingUpIterations} warming-up iterations.")

# Print the name of the GPU device
device = cuda.Device(0)  # Assuming you want to use the first GPU
print(f"Executing on {device.name()}")

# Compile the kernel code
mod = SourceModule(kernel_code)
compute_damages = mod.get_function("compute_damages")

# Simulation parameters
faultLineLength = 100

# Initialize data
magnitudes = np.random.uniform(low=2.0, high=7.0, size=numEvents).astype(np.float32)
epicenters = np.random.uniform(low=0.0, high=faultLineLength, size=numEvents).astype(np.float32)
locations = np.linspace(1, faultLineLength, numInsuredObjects, dtype=np.float32)
values = np.random.randint(300, 5000, size=numInsuredObjects).astype(np.float32)
types = np.random.randint(1, 4, size=numInsuredObjects).astype(np.int32)
soil_amp = np.random.randint(1, 4, size=numInsuredObjects).astype(np.float32)
losses = np.zeros(numEvents, dtype=np.float32)

# Allocate memory on the device
magnitudes_gpu = cuda.mem_alloc(magnitudes.nbytes)
epicenters_gpu = cuda.mem_alloc(epicenters.nbytes)
locations_gpu = cuda.mem_alloc(locations.nbytes)
values_gpu = cuda.mem_alloc(values.nbytes)
types_gpu = cuda.mem_alloc(types.nbytes)
soil_amp_gpu = cuda.mem_alloc(soil_amp.nbytes)
losses_gpu = cuda.mem_alloc(losses.nbytes)

# Copy data to the device
cuda.memcpy_htod(magnitudes_gpu, magnitudes)
cuda.memcpy_htod(epicenters_gpu, epicenters)
cuda.memcpy_htod(locations_gpu, locations)
cuda.memcpy_htod(values_gpu, values)
cuda.memcpy_htod(types_gpu, types)
cuda.memcpy_htod(soil_amp_gpu, soil_amp)
cuda.memcpy_htod(losses_gpu, losses)

# Set up the kernel execution configuration
threads_per_block = (16, 16)
blocks_per_grid_x = (numEvents + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (numInsuredObjects + threads_per_block[1] - 1) // threads_per_block[1]

# Warm-up the GPU
for _ in range(warmingUpIterations):
    compute_damages(magnitudes_gpu, epicenters_gpu, locations_gpu, values_gpu, types_gpu, soil_amp_gpu, losses_gpu, np.int32(numEvents), np.int32(numInsuredObjects), block=(16, 16, 1), grid=(numEvents // 16 + 1, numInsuredObjects // 16 + 1))

losses = np.zeros(numEvents, dtype=np.float32)
cuda.memcpy_htod(losses_gpu, losses)

# Start the timer
start_time = perf_counter()

# Run the kernel
compute_damages(magnitudes_gpu, epicenters_gpu, locations_gpu, values_gpu, types_gpu, soil_amp_gpu, losses_gpu, np.int32(numEvents), np.int32(numInsuredObjects), block=(threads_per_block[0], threads_per_block[1], 1), grid=(blocks_per_grid_x, blocks_per_grid_y))

# Stop the timer
cuda.Context.synchronize()  # Ensure all operations are finished
end_time = perf_counter()

print(f"GPU Computation Time: {(end_time - start_time)*1000:.6f} ms")

# Copy back the results and compute max and average
cuda.memcpy_dtoh(losses, losses_gpu)

# Sort the losses in descending order for further analysis
sorted_losses = np.sort(losses)[::-1]

max_loss = sorted_losses[0]
avg_loss = np.mean(losses)
print(f"Average loss: {avg_loss:.2f}")
print(f"Maximum loss: {max_loss:.2f}")
