clc; clear; close all;

trainPath = 'E:/study/MATH 3001 Projects/projects/experiment/Train';

imdsFull = imageDatastore(trainPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsVal] = splitEachLabel(imdsFull, 0.85, 'randomized');

inputSize = [227 227];

% Selective augmentation: simulate real-world perturbations
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize, imdsVal);

net = alexnet;
lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels));
newFc = fullyConnectedLayer(numClasses, ...
    'Name', 'fc_new', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
newClass = classificationLayer('Name', 'classoutput');

lgraph = replaceLayer(lgraph, 'fc8', newFc);
lgraph = replaceLayer(lgraph, 'output', newClass);

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 30);

[trainedNet, trainInfo] = trainNetwork(augTrain, lgraph, options);

predLabels = classify(trainedNet, augVal);
trueLabels = imdsVal.Labels;
accuracy = mean(predLabels == trueLabels);
fprintf('\nValidation Accuracy (Selective Aug Run5): %.2f%%\n', accuracy * 100);

% Save accuracy log
fid = fopen('selective_aug_run5_accuracy.txt', 'w');
fprintf(fid, 'Validation Accuracy: %.2f%%\n', accuracy * 100);
fclose(fid);

% Save model
save('selective_aug_run5_model.mat', 'trainedNet');

% Save training plot
f = figure('Visible', 'off');
plot(trainInfo.TrainingAccuracy, 'LineWidth', 2); hold on;
plot(trainInfo.TrainingLoss, 'LineWidth', 2);
legend('Accuracy', 'Loss');
xlabel('Epoch'); ylabel('Value');
title('Training Progress: Selective Aug Run5');
grid on;
saveas(f, 'selective_aug_run5_training_plot.png');
close(f);