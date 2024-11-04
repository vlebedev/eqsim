// Naive Earthquake Damage Calculation Model
//
// JCUDA implementation
// 
// Model assumptions:
// - All insured properties (assets) located on the EQ fault line and spead evenly
// - Each property has location, value, construction type and soil amplification parameters
// - Each EQ event has epicenter location and magnitude
//
// Damage function is defined as follows:
// damage = max(0, (1.0 - 0.01 * distance_from_epicenter) * (0.3 + 0.2 * construction_type + 0.02 * magnitude + 0.1 * amplification))
//
// Make sure it is opened in devcontainer, then run it with the following:
// # make all
// # mvn exec:java -Dexec.args="<numEvents> <numProperties> <warmingUpIterations>"

package com.example;

import static jcuda.driver.JCudaDriver.*;
import jcuda.*;
import jcuda.driver.*;

import java.util.Arrays;

public class EarthquakeSimulationJCuda {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Usage: mvn exec:java -Dexec.args=\"<numEvents> <numProperties> <warmingUpIterations>\"");
            System.exit(1);
        }

        int numEvents = Integer.parseInt(args[0]);
        int numInsuredObjects = Integer.parseInt(args[1]);
        int warmingUpIterations = Integer.parseInt(args[2]);

        System.out.println("Simulating earthquake damages for " + numEvents + " events and " + numInsuredObjects + " properties with " + warmingUpIterations + " warming-up iterations.");

        cuInit(0);
        CUdevice device = new CUdevice();
        CUcontext context = new CUcontext();
        checkResult(cuDeviceGet(device, 0), "cuDeviceGet");
        checkResult(cuCtxCreate(context, 0, device), "cuCtxCreate");

        // Get and print the device name
        byte[] deviceName = new byte[256];
        cuDeviceGetName(deviceName, 256, device);
        System.out.println("Executed on " + new String(deviceName).trim());

        float[] magnitudes = new float[numEvents];
        float[] epicenters = new float[numEvents];
        float[] locations = new float[numInsuredObjects];
        float[] values = new float[numInsuredObjects];
        int[] types = new int[numInsuredObjects];
        float[] soil_amp = new float[numInsuredObjects];
        float[] losses = new float[numEvents];

        initializeData(magnitudes, epicenters, locations, values, types, soil_amp, numEvents, numInsuredObjects);

        CUdeviceptr dMagnitudes = new CUdeviceptr();
        CUdeviceptr dEpicenters = new CUdeviceptr();
        CUdeviceptr dLocations = new CUdeviceptr();
        CUdeviceptr dValues = new CUdeviceptr();
        CUdeviceptr dTypes = new CUdeviceptr();
        CUdeviceptr dSoilAmp = new CUdeviceptr();
        CUdeviceptr dLosses = new CUdeviceptr();

        allocateAndCopyToDevice(dMagnitudes, dEpicenters, dLocations, dValues, dTypes, dSoilAmp, dLosses, magnitudes, epicenters, locations, values, types, soil_amp, numEvents, numInsuredObjects);

        CUmodule module = new CUmodule();
        checkResult(cuModuleLoad(module, "target/computeLosses.ptx"), "cuModuleLoad");
        CUfunction function = new CUfunction();
        checkResult(cuModuleGetFunction(function, module, "computeLosses"), "cuModuleGetFunction");

        Pointer kernelParameters = Pointer.to(
                Pointer.to(dMagnitudes),
                Pointer.to(dEpicenters),
                Pointer.to(dLocations),
                Pointer.to(dValues),
                Pointer.to(dTypes),
                Pointer.to(dSoilAmp),
                Pointer.to(dLosses),
                Pointer.to(new int[]{numEvents}),
                Pointer.to(new int[]{numInsuredObjects})
        );

        int blockSize = 256;
        int numBlocks = (numEvents + blockSize - 1) / blockSize;

        for (int i = 0; i < warmingUpIterations; i++) {
            checkResult(cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, null, kernelParameters, null), "Kernel Launch (Warming up)");
        }

        CUevent start = new CUevent();
        CUevent stop = new CUevent();
        cuEventCreate(start, 0);
        cuEventCreate(stop, 0);
        cuEventRecord(start, null);
        checkResult(cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, null, kernelParameters, null), "Kernel Launch");
        cuCtxSynchronize();
        cuEventRecord(stop, null);
        cuEventSynchronize(stop);

        float[] elapsedTime = new float[1];
        cuEventElapsedTime(elapsedTime, start, stop);
        cuEventDestroy(start);
        cuEventDestroy(stop);
        
        System.out.println("GPU Computation Time: " + elapsedTime[0] + " ms");

        cuMemcpyDtoH(Pointer.to(losses), dLosses, numEvents * Sizeof.FLOAT);
        Arrays.sort(losses);

        float maxLoss = losses[numEvents - 1];
        float sum = 0.0f;
        for (float loss : losses) {
            sum += loss;
        }
        float mean = sum / numEvents;
        System.out.println("Average loss: " + mean);
        System.out.println("Maximum loss: " + maxLoss);

        cleanup(context, dMagnitudes, dEpicenters, dLocations, dValues, dTypes, dSoilAmp, dLosses);
    }

    private static void initializeData(float[] magnitudes, float[] epicenters, float[] locations, float[] values, int[] types, float[] soil_amp, int numEvents, int numInsuredObjects) {
        for (int i = 0; i < numEvents; i++) {
            magnitudes[i] = 2 + (float)Math.random() * 5;
            epicenters[i] = (float)Math.random() * 100;
        }
        for (int i = 0; i < numInsuredObjects; i++) {
            locations[i] = 1.0f + 99.0f * i / (numInsuredObjects - 1);
            values[i] = 300 + (float)Math.random() * 4700;
            types[i] = 1 + (int)(Math.random() * 3);
            soil_amp[i] = 1 + (float)Math.random() * 3;
        }
    }

    private static void allocateAndCopyToDevice(CUdeviceptr dMagnitudes, CUdeviceptr dEpicenters, CUdeviceptr dLocations, CUdeviceptr dValues, CUdeviceptr dTypes, CUdeviceptr dSoilAmp, CUdeviceptr dLosses, float[] magnitudes, float[] epicenters, float[] locations, float[] values, int[] types, float[] soil_amp, int numEvents, int numInsuredObjects) {
        cuMemAlloc(dMagnitudes, numEvents * Sizeof.FLOAT);
        cuMemAlloc(dEpicenters, numEvents * Sizeof.FLOAT);
        cuMemAlloc(dLocations, numInsuredObjects * Sizeof.FLOAT);
        cuMemAlloc(dValues, numInsuredObjects * Sizeof.FLOAT);
        cuMemAlloc(dTypes, numInsuredObjects * Sizeof.INT);
        cuMemAlloc(dSoilAmp, numInsuredObjects * Sizeof.FLOAT);
        cuMemAlloc(dLosses, numEvents * Sizeof.FLOAT);

        cuMemcpyHtoD(dMagnitudes, Pointer.to(magnitudes), numEvents * Sizeof.FLOAT);
        cuMemcpyHtoD(dEpicenters, Pointer.to(epicenters), numEvents * Sizeof.FLOAT);
        cuMemcpyHtoD(dLocations, Pointer.to(locations), numInsuredObjects * Sizeof.FLOAT);
        cuMemcpyHtoD(dValues, Pointer.to(values), numInsuredObjects * Sizeof.FLOAT);
        cuMemcpyHtoD(dTypes, Pointer.to(types), numInsuredObjects * Sizeof.INT);
        cuMemcpyHtoD(dSoilAmp, Pointer.to(soil_amp), numInsuredObjects * Sizeof.FLOAT);
    }

    private static void cleanup(CUcontext context, CUdeviceptr... resources) {
        for (CUdeviceptr resource : resources) {
            cuMemFree(resource);
        }
        cuCtxDestroy(context);
    }

    private static void checkResult(int result, String action) {
        if (result != CUresult.CUDA_SUCCESS) {
            System.err.println(action + " failed: " + CUresult.stringFor(result));
            System.exit(1);
        }
    }
}
