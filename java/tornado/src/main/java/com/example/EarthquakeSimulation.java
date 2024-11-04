// Naive Earthquake Damage Calculation Model
//
// TornadoVM implementation
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
// # mvn clean
// # mvn package
// # tornado -cp target/example-1.0-SNAPSHOT.jar com.example.EarthquakeSimulation <numEvents> <numProperties> <warmingUpIterations>
 
package com.example;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class EarthquakeSimulation {

    private static void computeDamages(FloatArray magnitudes, FloatArray epicenters, FloatArray locations, FloatArray values, FloatArray types, 
                                       FloatArray soilAmp, FloatArray losses, final int numEvents, final int numProperties) {
        for (@Parallel int i = 0; i < numEvents; i++) {
            float totalLoss = 0f;
            float magnitude = magnitudes.get(i);
            float epicenter = epicenters.get(i);
            
            for (int j = 0; j < numProperties; j++) {

                float distance = Math.abs(locations.get(j) - epicenter);
                float type = types.get(j);
                float amp = soilAmp.get(j);
                float damage = Math.max(0f, (1.0f - 0.01f * distance) * (0.3f + 0.2f * type + 0.02f * magnitude + 0.1f * amp));
                
                totalLoss += damage * values.get(j);
            }

            losses.set(i, totalLoss);
        }
    }

    private static void computeDamagesParallel(FloatArray magnitudes, FloatArray epicenters, FloatArray locations, FloatArray values, FloatArray types, 
                                               FloatArray soilAmp, FloatArray losses, final int numEvents, final int numProperties) {
        // Parallel stream for handling events
        IntStream.range(0, numEvents).parallel().forEach(i -> {
            float totalLoss = 0f;
            float magnitude = magnitudes.get(i);
            float epicenter = epicenters.get(i);
    
            for (int j = 0; j < numProperties; j++) {
                float distance = Math.abs(locations.get(j) - epicenter);
                float type = types.get(j);
                float amp = soilAmp.get(j);
                float damage = Math.max(0f, (1.0f - 0.01f * distance) * (0.3f + 0.2f * type + 0.02f * magnitude + 0.1f * amp));
                totalLoss += damage * values.get(j);
            }
    
            losses.set(i, totalLoss);
        });
    }
  
    public static void main(String[] args) {

        if (args.length < 3) {
            System.out.println("Usage: tornado -cp target/example-1.0-SNAPSHOT.jar com.example.EarthquakeSimulation <numEvents> <numProperties> <warmingUpIterations>");
            System.exit(1);
        }

        int numEvents = Integer.parseInt(args[0]);
        int numProperties = Integer.parseInt(args[1]);
        int warmingUpIterations = Integer.parseInt(args[2]);

        System.out.println("Simulating earthquake damages for " + numEvents + " events and " + numProperties + " properties with " + warmingUpIterations + " warming-up iterations.");

        FloatArray magnitudes = new FloatArray(numEvents);
        FloatArray epicenters = new FloatArray(numEvents);
        FloatArray losses = new FloatArray(numEvents);

        FloatArray locations = new FloatArray(numProperties);
        FloatArray values = new FloatArray(numProperties);
        FloatArray types = new FloatArray(numProperties);
        FloatArray soilAmp = new FloatArray(numProperties); 

        Random r = new Random(123);
        
        for (int i = 0; i < numEvents; i++) {
            magnitudes.set(i, 2 + 5 * r.nextFloat()); // Random magnitudes between 2 and 7
            epicenters.set(i, 100 * r.nextFloat()); // Random epicenter locations between 0 and 100
        }

        for (int i = 0; i < numProperties; i++) {
           locations.set(i, 1 + 99 * i / (numProperties - 1)); // Linearly spaced locations
           values.set(i, 300 + 4700 * r.nextFloat()); // Random values between 300K and 5000K
           types.set(i, (float) r.nextInt(3) + 1); // Random types between 1 and 3
           soilAmp.set(i, (float) r.nextInt(3) + 1); // Random soil amplification factors between 1 and 3
        }

        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, magnitudes, epicenters, locations, values, types, soilAmp)
                .task("t0", EarthquakeSimulation::computeDamages, magnitudes, epicenters, locations, values, types, soilAmp, losses, numEvents, numProperties)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, losses);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);

        // 1. Warm up CPU
        for (int i = 0; i < warmingUpIterations; i++) {
            computeDamages(magnitudes, epicenters, locations, values, types, soilAmp, losses, numEvents, numProperties);
        }

        // 2. Perform sequential CPU computation
        long startCPU = System.nanoTime();
        computeDamages(magnitudes, epicenters, locations, values, types, soilAmp, losses, numEvents, numProperties);
        long endCPU = System.nanoTime();

        // 3. Warm up CPU using parallel computation
        for (int i = 0; i < warmingUpIterations; i++) {
            computeDamagesParallel(magnitudes, epicenters, locations, values, types, soilAmp, losses, numEvents, numProperties);
        }

        // 4. Perform parallel CPU computation
        long startCPUp = System.nanoTime();
        computeDamagesParallel(magnitudes, epicenters, locations, values, types, soilAmp, losses, numEvents, numProperties);
        long endCPUp = System.nanoTime();

        // 5. Warm up GPU with TornadoVM
        for (int i = 0; i < warmingUpIterations; i++) {
            executor.execute();
        }

        // 6. Measure GPU execution time
        long startGPU = System.nanoTime();
        executor.execute();
        long endGPU = System.nanoTime();

        float[] h_losses = losses.toHeapArray();
        
        Arrays.sort(h_losses);

        // Reverse the array to make it descending
        for (int i = 0; i < h_losses.length / 2; i++) {
            float temp = h_losses[i];
            h_losses[i] = h_losses[h_losses.length - 1 - i];
            h_losses[h_losses.length - 1 - i] = temp;
        }
        
        float sum = 0;

        for (float value : h_losses) {
            sum += value;
        }

        String deviceName = executor.getDevice(0).getDescription();
        System.out.println("Executed on " + deviceName);
        System.out.println("Average loss: " + sum / h_losses.length);
        System.out.println("Maximum loss: " + h_losses[0]);
        System.out.println("CPU Computation Time (single thread): " + (endCPU - startCPU) / 1000000f + " ms");
        System.out.println("CPU Computation Time (parallel):      " + (endCPUp - startCPUp) / 1000000f + " ms");
        System.out.println("GPU Computation Time:                 " + (endGPU - startGPU) / 1000000f + " ms");
    }
}
