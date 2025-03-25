clc; clear; close all;

valPath = 'E:/study/MATH 3001 Projects/projects/experiment/RobustnessTest/mid';

imdsVal = imageDatastore(valPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

inputSize = [227 227];
augVal = augmentedImageDatastore(inputSize, imdsVal);

load('aug_adv_run2_model.mat', 'trainedNet');

predLabels = classify(trainedNet, augVal);
trueLabels = imdsVal.Labels;

accuracy = mean(predLabels == trueLabels);
fprintf('\nRobustness Test Accuracy (Aug Advanced Run2): %.2f%%\n', accuracy * 100);

fid = fopen('aug_adv_robustness_run2_mid_accuracy.txt', 'w');
fprintf(fid, 'Robustness Test Accuracy (Aug Advanced Run2): %.2f%%\n', accuracy * 100);
fclose(fid);