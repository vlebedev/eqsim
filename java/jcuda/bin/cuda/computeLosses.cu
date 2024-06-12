extern "C"
__global__ void computeLosses(float *magnitudes, float *epicenters, float *locations, float *values, int *types, float *soil_amp,
                              float *losses, int numEvents, int numInsuredObjects) {
    int eventIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eventIdx < numEvents) {
        float totalLoss = 0.0;
        for (int i = 0; i < numInsuredObjects; i++) {
            float distance = fabs(locations[i] - epicenters[eventIdx]);
            float damage = max(0.0, (1.0 - 0.01 * distance) * (0.3 + 0.2 * types[i] + 0.02 * magnitudes[eventIdx] + 0.1 * soil_amp[i]));
            totalLoss += damage * values[i];
        }
        losses[eventIdx] = totalLoss;
    }
}
