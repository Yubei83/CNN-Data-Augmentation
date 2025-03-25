clc; clear; close all;

trainPath = 'E:/study/MATH 3001 Projects/projects/experiment/Train';

imdsFull = imageDatastore(trainPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsVal] = splitEachLabel(imdsFull, 0.85, 'randomized');

inputSize = [227 227];

augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

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
fprintf('\nValidation Accuracy (Aug Advanced Run4): %.2f%%\n', accuracy * 100);

fid = fopen('aug_adv_run4_accuracy.txt', 'w');
fprintf(fid, 'Validation Accuracy: %.2f%%\n', accuracy * 100);
fclose(fid);

save('aug_adv_run4_model.mat', 'trainedNet');

f = figure('Visible', 'off');
plot(trainInfo.TrainingAccuracy, 'LineWidth', 2); hold on;
plot(trainInfo.TrainingLoss, 'LineWidth', 2);
legend('Accuracy', 'Loss');
xlabel('Epoch'); ylabel('Value');
title('Training Progress: Aug Adv Run4');
grid on;
saveas(f, 'aug_adv_run4_training_plot.png');
close(f);
。、89