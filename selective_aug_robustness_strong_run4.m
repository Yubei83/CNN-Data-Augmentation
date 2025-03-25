clc; clear; close all;

valPath = 'E:/study/MATH 3001 Projects/projects/experiment/RobustnessTest/strong';

imdsVal = imageDatastore(valPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

inputSize = [227 227];
augVal = augmentedImageDatastore(inputSize, imdsVal);

load('selective_aug_run4_model.mat', 'trainedNet');

predLabels = classify(trainedNet, augVal);
trueLabels = imdsVal.Labels;

accuracy = mean(predLabels == trueLabels);
fprintf('\nRobustness Test Accuracy (Selective Aug Run4): %.2f%%\n', accuracy * 100);

% Save accuracy log
fid = fopen('selective_aug_robustness_accuracy.txt', 'w');
fprintf(fid, 'Robustness Test Accuracy (Selective Aug Run4): %.2f%%\n', accuracy * 100);
fclose(fid);