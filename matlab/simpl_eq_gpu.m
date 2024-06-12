% Naive Earthquake Damage Calculation Model
%
% Matlab (GPU, vectorized) implementation
% 
% Model assumptions:
% - All insured properties (assets) located on the EQ fault line and spead evenly
% - Each property has location, value, construction type and soil amplification parameters
% - Each EQ event has epicenter location and magnitude
%
% Damage function is defined as follows:
% damage = max(0, (1.0 - 0.01 * distance_from_epicenter) * (0.3 + 0.2 * construction_type + 0.02 * magnitude + 0.1 * amplification))

function losses = calculateDamages(magnitudes, epicenters, insuredObjects)
    % Extract individual arrays from the insuredObjects matrix
    locations = insuredObjects(:, 1);
    values = insuredObjects(:, 2);
    types = insuredObjects(:, 3);
    soilAmp = insuredObjects(:, 4);
    
    % Expand arrays to match dimensions for vectorized operations
    magMatrix = repmat(magnitudes, 1, numel(locations));
    epicMatrix = repmat(epicenters, 1, numel(locations));
    locMatrix = repmat(locations', numel(magnitudes), 1);
    valMatrix = repmat(values', numel(magnitudes), 1);
    typeMatrix = repmat(types', numel(magnitudes), 1);
    soilAmpMatrix = repmat(soilAmp', numel(magnitudes), 1);

    % Calculate distances and damage ratios in a vectorized manner
    distances = abs(locMatrix - epicMatrix);
    damageRatios = max(0, (1 - 0.01 * distances) .* (0.3 + 0.2 * typeMatrix + 0.02 * magMatrix + 0.1 * soilAmpMatrix));
    lossesMatrix = damageRatios .* valMatrix;

    % Sum up the losses for each event
    losses = sum(lossesMatrix, 2);
end

% Main Script
numInsuredObjects = 1000;
numEvents = 100000;
faultLineLength = 100;
minMagnitude = 2; 
maxMagnitude = 7;

warmingUpIterations = 0;

% Generate data
objectsCoordinates = linspace(1, faultLineLength, numInsuredObjects)';
insuredObjects = [objectsCoordinates, randi([300, 5000], numInsuredObjects, 1), ...
    randi([1, 3], numInsuredObjects, 1), randi([1, 3], numInsuredObjects, 1)];
magnitudes = minMagnitude + rand(numEvents, 1) * (maxMagnitude - minMagnitude);
epicenters = rand(numEvents, 1) * faultLineLength;

% Convert data to GPU arrays
gpuInsuredObjects = gpuArray(insuredObjects);
gpuMagnitudes = gpuArray(magnitudes);
gpuEpicenters = gpuArray(epicenters);

% Warm-up the GPU
for i=1:warmingUpIterations
    tmp = calculateDamages(gpuMagnitudes, gpuEpicenters, gpuInsuredObjects);
end

% Calculate damages using GPU
tic;
gpuLosses = calculateDamages(gpuMagnitudes, gpuEpicenters, gpuInsuredObjects);
losses = gather(gpuLosses);  % Retrieve results from GPU to CPU
toc;

% Output results
meanLoss = mean(losses);
sortedLosses = sort(losses, 'descend');
fprintf('Average loss: %.2f\n', meanLoss);
fprintf('Maximum loss: %.2f\n', sortedLosses(1));
