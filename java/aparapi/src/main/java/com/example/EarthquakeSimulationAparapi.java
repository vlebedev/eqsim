// Naive Earthquake Damage Calculation Model
//
// Aparapi implementation
// 
// Model assumptions:
// - All insured properties (assets) located on the EQ fault line and spread evenly
// - Each property has location, value, construction type and soil amplification parameters
// - Each EQ event has epicenter location and magnitude
//
// Damage function is defined as follows:
// damage = max(0, (1.0 - 0.01 * distance_from_epicenter) * (0.3 + 0.2 * construction_type + 0.02 * magnitude + 0.1 * amplification))
//
// Make sure it is opened in devcontainer, then run it with the following:
// # mvn exec:java -Dexec.args="<numEvents> <numProperties> <warmingUpIterations>"

package com.example;

import com.aparapi.Kernel;
import com.aparapi.Range;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import static jcuda.driver.JCudaDriver.*;

import java.util.Arrays;

public class EarthquakeSimulationAparapi {
    static {
        setExceptionsEnabled(true);
        cuInit(0);
    }

    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Usage: mvn exec:java -Dexec.args=\"<numEvents> <numProperties> <warmingUpIterations>\"");
            System.exit(1);
        }

        int numEvents = Integer.parseInt(args[0]);
        int numInsuredObjects = Integer.parseInt(args[1]);
        int warmingUpIterations = Integer.parseInt(args[2]);

        CUdevice device = new CUdevice();
        CUcontext context = new CUcontext();
        cuDeviceGet(device, 0); // Get the first CUDA device
        cuCtxCreate(context, 0, device);

        // Get and print the device name
        byte[] deviceName = new byte[256];
        cuDeviceGetName(deviceName, 256, device);
        System.out.println("Executed on " + new String(deviceName).trim());

        // Initialize data arrays
        final float[] magnitudes = new float[numEvents];
        final float[] epicenters = new float[numEvents];
        final float[] locations = new float[numInsuredObjects];
        final float[] values = new float[numInsuredObjects];
        final int[] types = new int[numInsuredObjects];
        final float[] soil_amp = new float[numInsuredObjects];
        final float[] losses = new float[numEvents];

        // Populate data arrays
        for (int i = 0; i < numEvents; i++) {
            magnitudes[i] = 2 + (float)Math.random() * 5;
            epicenters[i] = (float)Math.random() * 100;
        }
        for (int i = 0; i < numInsuredObjects; i++) {
            locations[i] = 1 + 99 * i / (numInsuredObjects - 1);
            values[i] = 300 + (float)Math.random() * 4700;
            types[i] = 1 + (int)(Math.random() * 3);
            soil_amp[i] = 1 + (float)Math.random() * 3;
        }

        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int eventId = getGlobalId();
                float totalLoss = 0;
                float distance, damage;
                for (int i = 0; i < numInsuredObjects; i++) {
                    distance = abs(locations[i] - epicenters[eventId]);
                    damage = max(0, (1 - 0.01f * distance) * (0.3f + 0.2f * types[i] + 0.02f * magnitudes[eventId] + 0.1f * soil_amp[i]));
                    totalLoss += damage * values[i];
                }
                losses[eventId] = totalLoss;
            }
        };

        for (int i = 0; i < warmingUpIterations; i++) {
            kernel.execute(Range.create(numEvents));    
        }

        long startTime = System.nanoTime();
        kernel.execute(Range.create(numEvents));
        long endTime = System.nanoTime();

        Arrays.sort(losses);
        for (int i = 0; i < losses.length / 2; i++) {
            float temp = losses[i];
            losses[i] = losses[losses.length - 1 - i];
            losses[losses.length - 1 - i] = temp;
        }

        float maxLoss = losses[0];
        float totalLoss = 0;
        for (float loss : losses) {
            totalLoss += loss;
        }
        float avgLoss = totalLoss / numEvents;

        System.out.println("GPU Computation Time: " + (endTime - startTime) / 1e6 + " ms");
        System.out.println("Average loss: " + avgLoss);
        System.out.println("Maximum loss: " + maxLoss);

        kernel.dispose();
        cuCtxDestroy(context);
        System.exit(0);
    }
}
