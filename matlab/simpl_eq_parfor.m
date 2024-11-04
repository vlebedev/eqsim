% Naive Earthquake Damage Calculation Model
%
% Matlab (CPU, multi-threaded) implementation
% 
% Model assumptions:
% - All insured properties (assets) located on the EQ fault line and spead evenly
% - Each property has location, value, construction type and soil amplification parameters
% - Each EQ event has epicenter location and magnitude
%
% Damage function is defined as follows:
% damage = max(0, (1.0 - 0.01 * distance_from_epicenter) * (0.3 + 0.2 * construction_type + 0.02 * magnitude + 0.1 * amplification))

% Simulation parameters
numInsuredObjects = 1000;
numEvents = 100000;
faultLineLength = 100;
minMagnitude = 2; 
maxMagnitude = 7;

warmingUpIterations = 0;

function losses = calculateDamages(magnitudes, epicenters, insuredObjects, numEvents)

    damageRatio = @(distance, type, magnitude, soilAmp) max(0, (1 - 0.01*distance) .* (0.3 + 0.2*type + 0.02*magnitude + 0.1*soilAmp));
    losses = zeros(numEvents, 1);

    parfor i=1:numEvents
    
    % Current event properties
        magnitude = magnitudes(i);
        epicenter = epicenters(i);
    
        % Calculate damage for each object
        totalLoss = 0;
        for j = 1:size(insuredObjects, 1)
            object = insuredObjects(j,:);
            distance = abs(object(1) - epicenter); % Simple distance to epicenter
            propertyType = object(3);
            damage = damageRatio(distance, propertyType, magnitude, object(4));
    
            % Calculate financial loss based on value and damage ratio
            loss = damage * object(2);
            totalLoss = totalLoss + loss;
        end
    
        % Record the total loss for this simulation
        losses(i) = totalLoss;
    
    end
end

function losses = calculateDamagesOptimized(magnitudes, epicenters, insuredObjects, numEvents, numWorkers)
    numInsuredObjects = size(insuredObjects, 1);
    
    % Pre-allocate losses array
    losses = zeros(numEvents, 1);
    
    % Calculate chunk size for each worker
    chunkSize = ceil(numEvents / numWorkers);
    
    % Temporary storage for each chunk's results
    chunkLosses = cell(numWorkers, 1);
    
    parfor i = 1:numWorkers
        startIdx = (i - 1) * chunkSize + 1;
        endIdx = min(i * chunkSize, numEvents);
        tempLosses = zeros(endIdx - startIdx + 1, 1);
        
        % Process each event in the chunk
        for j = startIdx:endIdx
            magnitude = magnitudes(j);
            epicenter = epicenters(j);
            totalLoss = 0;
            
            for k = 1:numInsuredObjects
                object = insuredObjects(k,:);
                distance = abs(object(1) - epicenter);
                damage = max(0, (1 - 0.01 * distance) * (0.3 + 0.2 * object(3) + 0.02 * magnitude + 0.1 * object(4)));
                totalLoss = totalLoss + damage * object(2);
            end
            tempLosses(j - startIdx + 1) = totalLoss;
        end
        
        % Store the temporary results in the cell array
        chunkLosses{i} = tempLosses;
    end
    
    % Consolidate results from each chunk
    for i = 1:numWorkers
        startIdx = (i - 1) * chunkSize + 1;
        endIdx = min(i * chunkSize, numEvents);
        losses(startIdx:endIdx) = chunkLosses{i};
    end
end

% Define portfolio data: location, value, construction type
% All objects are in a line along a fault line for simplicity
objectsCoordinates = linspace(1, faultLineLength, numInsuredObjects)';
insuredObjects = [objectsCoordinates, randi([300,5000], numInsuredObjects, 1), ...
    randi([1,3], numInsuredObjects, 1), randi([1,3], numInsuredObjects, 1)]; % [Location, Value ($K), Construction Type, Soil Amplification]

% Randomly generate EQ events (magnitudes and epicenter locations)
magnitudes = minMagnitude + rand(numEvents, 1) * (maxMagnitude - minMagnitude); % Magnitudes between minMagnitude and maxMagnitude
epicenters = rand(numEvents, 1) * faultLineLength;  % Epicenter locations between 0 and faultLineLen

numWorkers = feature('numcores');  % Adapting to the number of physical cores

if isempty(gcp('nocreate'))
    parpool(numWorkers);  % Start a parallel pool using default settings
end

% Warming-up parpool

for i=1:warmingUpIterations
    tmp = calculateDamages(magnitudes, epicenters, insuredObjects, numEvents);
end

% Calculate Total Event Losses across all insured objects

tic;
losses = calculateDamagesOptimized(magnitudes, epicenters, insuredObjects, numEvents, numWorkers);
toc;

% Analyze the results
meanLoss = mean(losses);

sortedLosses = sort(losses, 'descend');

fprintf('Average loss: %.2f\n', meanLoss);
fprintf('Maximum loss: %.2f\n', sortedLosses(1));
