%% Experiment Code for AlexNet on CIFAR-10 using Combined Datastores
% This code loads CIFAR-10 data, creates combined datastores that return both images and labels,
% applies resizing (and optional augmentation), modifies the untrained AlexNet for 10-class classification,
% and trains the network.

clear; clc; close all;

%% 1. Load CIFAR-10 Data
% Load training batches
batch1 = load('data_batch_1.mat');
batch2 = load('data_batch_2.mat');
batch3 = load('data_batch_3.mat');
batch4 = load('data_batch_4.mat');
batch5 = load('data_batch_5.mat');
% Load test batch
testBatch = load('test_batch.mat');

% Combine training data and labels
XTrain = double(cat(1, batch1.data, batch2.data, batch3.data, batch4.data, batch5.data));
YTrain = categorical(cat(1, batch1.labels, batch2.labels, batch3.labels, batch4.labels, batch5.labels));
XTest = double(testBatch.data);
YTest = categorical(testBatch.labels);

% Reshape from [numSamples, 3072] to [32, 32, 3, numSamples]
XTrain = reshape(XTrain', [32, 32, 3, size(XTrain, 1)]);
XTest = reshape(XTest', [32, 32, 3, size(XTest, 1)]);

% Permute dimensions
XTrain = permute(XTrain, [2, 1, 3, 4]);
XTest = permute(XTest, [2, 1, 3, 4]);

% Normalize pixel values to [0, 1]
XTrain = XTrain / 255;
XTest = XTest / 255;

%% 2. Create Combined Datastores with Resizing and Augmentation
% Define input size for AlexNet
inputSize = [227 227 3];

% Create array datastores for images and labels for training
dsXTrain = arrayDatastore(XTrain, 'IterationDimension', 4);
dsYTrain = arrayDatastore(YTrain);
combinedTrainDS = combine(dsXTrain, dsYTrain);

% Create combined datastore for test data
dsXTest = arrayDatastore(XTest, 'IterationDimension', 4);
dsYTest = arrayDatastore(YTest);
combinedTestDS = combine(dsXTest, dsYTest);

% Define a function to resize images
resizeFcn = @(data) { imresize(data{1}, inputSize(1:2)), data{2} };

% Create baseline datastores (only resizing)
baselineTrainDS = transform(combinedTrainDS, resizeFcn);
baselineTestDS = transform(combinedTestDS, resizeFcn);

% Define data augmentation parameters
augmenter = imageDataAugmenter(...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5], ...
    'RandXScale', [0.9, 1.1], ...
    'RandYScale', [0.9, 1.1], ...
    'RandXReflection', true);
% Create augmented training datastore with resizing and augmentation
augFcn = @(data) { augment(augmenter, imresize(data{1}, inputSize(1:2))), data{2} };
augmentedTrainDS = transform(combinedTrainDS, augFcn);

%% 3. Load and Modify AlexNet
% Load untrained AlexNet to avoid the need for the support package
net = imagePretrainedNetwork('alexnet', 'Weights', 'none');
lgraph = layerGraph(net);

% Replace the final fully connected layer ('fc8') to output 10 classes
newFc = fullyConnectedLayer(10, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, 'fc8', newFc);

% Replace the softmax layer ('prob') with a new softmax layer
newSoftmax = softmaxLayer('Name', 'new_softmax');
lgraph = replaceLayer(lgraph, 'prob', newSoftmax);

% Check for the existence of a classification layer and modify accordingly
if ismember('ClassificationLayer_fc8', {lgraph.Layers.Name})
    newClassLayer = classificationLayer('Name', 'new_classoutput');
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc8', newClassLayer);
else
    newClassLayer = classificationLayer('Name', 'new_classoutput');
    lgraph = addLayers(lgraph, newClassLayer);
    lgraph = connectLayers(lgraph, 'new_softmax', 'new_classoutput');
end

% Verify network structure; there should be no missing or unconnected layers
analyzeNetwork(lgraph);

%% 4. Set Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...          % Adjust epochs as needed
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'gpu');

%% 5. Train the Network (Baseline and Augmented Models)
fprintf('Training baseline model...\n');
trainedNetBaseline = trainNetwork(baselineTrainDS, lgraph, options);

fprintf('Training augmented model...\n');
trainedNetAugmented = trainNetwork(augmentedTrainDS, lgraph, options);

%% 6. Evaluate the Models on Test Data
YPredBaseline = classify(trainedNetBaseline, baselineTestDS);
accuracyBaseline = mean(YPredBaseline == YTest);
fprintf('Baseline Test Accuracy: %.2f%%\n', accuracyBaseline * 100);

YPredAugmented = classify(trainedNetAugmented, baselineTestDS);
accuracyAugmented = mean(YPredAugmented == YTest);
fprintf('Augmented Test Accuracy: %.2f%%\n', accuracyAugmented * 100);
