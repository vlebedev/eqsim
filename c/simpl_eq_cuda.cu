// Naive Earthquake Damage Calculation Model
//
// C/CUDA implementation
// 
// Model assumptions:
// - All insured properties (assets) all located on the EQ fault line and spead evenly
// - Each property has location, value, construction type and soil amplification parameters
// - Each EQ event has epicenter location and magnitude
//
// Damage function is defined as follows:
// damage = Math.max(0, (1.0 - 0.01 * distance_from_epicenter) * (0.3 + 0.2 * construction_type + 0.02 * magnitude + 0.1 * amplification))
//
// Make sure it is opened in devcontainer, then run it with the following:
// # nvcc -o simpl_eq_cuda simpl_eq_cuda.cu
// # ./simpl_eq_cuda <numEvents> <numProperties> <warmingUpIterations>

#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <math.h>

__global__ void computeLosses(float *magnitudes, float *epicenters, float *locations, float *values, float *types, float *soilAmp, float *losses, int numEvents, int numInsuredObjects) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEvents) {
        float totalLoss = 0.0;
        float magnitude = magnitudes[idx];
        float epicenter = epicenters[idx];

        for (int j = 0; j < numInsuredObjects; j++) {
            float distance = fabs(locations[j] - epicenter);
            int type = types[j];
            float damage = fmax(0.0, (1.0 - 0.01 * distance) * (0.3 + 0.2 * type + 0.02 * magnitude + 0.1 * soilAmp[j]));
            totalLoss += damage * values[j];
        }

        losses[idx] = totalLoss;
    }
}

int main(int argc, char *argv[]) {

    if (argc < 4) {
        printf("Usage: %s <numEvents> <numInsuredObjects> <warmingUpIterations>\n", argv[0]);
        return 1;
    }

    int numEvents = atoi(argv[1]);
    int numInsuredObjects = atoi(argv[2]);
    int warmingUpIterations = atoi(argv[3]);

    printf("\nSimulating earthquake damages for %d events and %d properties with %d warming-up iterations.\n", 
           numEvents, numInsuredObjects, warmingUpIterations);

    // Allocate and initialize data on the host
    float *h_magnitudes = (float *)malloc(numEvents * sizeof(float));
    float *h_epicenters = (float *)malloc(numEvents * sizeof(float));
    float *h_locations = (float *)malloc(numInsuredObjects * sizeof(float));
    float *h_values = (float *)malloc(numInsuredObjects * sizeof(float));
    float *h_types = (float *)malloc(numInsuredObjects * sizeof(float));
    float *h_soilAmp = (float *)malloc(numInsuredObjects * sizeof(float));
    float *h_losses = (float *)malloc(numEvents * sizeof(float));

    // Initialize data
    for (int i = 0; i < numEvents; i++) {
        h_magnitudes[i] = 2.0 + (7.0 - 2.0) * rand() / RAND_MAX; // Random magnitudes between 2 and 7
        h_epicenters[i] = 100.0 * rand() / RAND_MAX; // Random epicenter locations between 0 and 100
    }
    for (int i = 0; i < numInsuredObjects; i++) {
        h_locations[i] = 1.0 + 99.0 * i / (numInsuredObjects - 1); // Linearly spaced locations
        h_values[i] = 300.0 + (5000.0 - 300.0) * rand() / RAND_MAX; // Random values between 300K and 5000K
        h_types[i] = rand() % 3 + 1; // Random types between 1 and 3
        h_soilAmp[i] = rand() % 3 + 1; // Random soil amplification factors between 1 and 3
    }

    // Device memory pointers
    float *d_magnitudes, *d_epicenters, *d_locations, *d_values, *d_types, *d_soilAmp;
    float *d_losses;

    // Allocate memory on the device
    cudaMalloc(&d_magnitudes, numEvents * sizeof(float));
    cudaMalloc(&d_epicenters, numEvents * sizeof(float));
    cudaMalloc(&d_locations, numInsuredObjects * sizeof(float));
    cudaMalloc(&d_values, numInsuredObjects * sizeof(float));
    cudaMalloc(&d_types, numInsuredObjects * sizeof(float));
    cudaMalloc(&d_soilAmp, numInsuredObjects * sizeof(float));
    cudaMalloc(&d_losses, numEvents * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_magnitudes, h_magnitudes, numEvents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_epicenters, h_epicenters, numEvents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_locations, h_locations, numInsuredObjects * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, numInsuredObjects * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_types, h_types, numInsuredObjects * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_soilAmp, h_soilAmp, numInsuredObjects * sizeof(float), cudaMemcpyHostToDevice);

    // Get device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    // Set blockSize based on the GPU capabilities
    int blockSize = properties.maxThreadsPerBlock;

    // Adjust blockSize if necessary (optional, based on your specific needs)
    if (blockSize > 256) {
        blockSize = 256; // Example: limit blockSize to 256 for better efficiency
    }

    int numBlocks = (numEvents + blockSize - 1) / blockSize;

    for (int i = 0; i < warmingUpIterations; i++) {
        computeLosses<<<numBlocks, blockSize>>>(d_magnitudes, d_epicenters, d_locations, d_values, d_types, d_soilAmp, d_losses, numEvents, numInsuredObjects);
        cudaDeviceSynchronize();
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Launch the kernel with optimized blockSize
    computeLosses<<<numBlocks, blockSize>>>(d_magnitudes, d_epicenters, d_locations, d_values, d_types, d_soilAmp, d_losses, numEvents, numInsuredObjects);
    cudaDeviceSynchronize();

    // Stop timing after the GPU work is done
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Sort losses using Thrust
    thrust::device_ptr<float> d_losses_ptr(d_losses);
    thrust::sort(thrust::device, d_losses_ptr, d_losses_ptr + numEvents, thrust::greater<float>());

    // Calculate the sum using thrust::reduce
    float sum = thrust::reduce(thrust::device, d_losses_ptr, d_losses_ptr + numEvents);
    float mean = sum / numEvents;
   
    // Copy results back to host
    cudaMemcpy(h_losses, d_losses, numEvents * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Executed on %s\n", properties.name);
    printf("Average Loss: %f\n", mean);
    printf("Maximum Loss: %f\n", h_losses[0]);
    printf("GPU Computation time: %f ms\n", milliseconds);
    
    // Free device memory
    cudaFree(d_magnitudes);
    cudaFree(d_epicenters);
    cudaFree(d_locations);
    cudaFree(d_values);
    cudaFree(d_types);
    cudaFree(d_soilAmp);
    cudaFree(d_losses);

    // Free host memory
    free(h_magnitudes);
    free(h_epicenters);
    free(h_locations);
    free(h_values);
    free(h_types);
    free(h_soilAmp);
    free(h_losses);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
