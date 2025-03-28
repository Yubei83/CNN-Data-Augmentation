clc; clear; close all;

valPath = 'E:/study/MATH 3001 Projects/projects/experiment/RobustnessTest/strong';

imdsVal = imageDatastore(valPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

inputSize = [227 227];
augVal = augmentedImageDatastore(inputSize, imdsVal);

load('aug_basic_run1_model.mat', 'trainedNet');

predLabels = classify(trainedNet, augVal);
trueLabels = imdsVal.Labels;

accuracy = mean(predLabels == trueLabels);
fprintf('\nRobustness Test Accuracy (Aug Basic Run1): %.2f%%\n', accuracy * 100);

fid = fopen('aug_basic_robustness_strong_run1_accuracy.txt', 'w');
fprintf(fid, 'Robustness Test Accuracy (Aug Basic Run1): %.2f%%\n', accuracy * 100);
fclose(fid);