clc; clear; close all;

valPath = ['E:/study/MATH 3001 Projects/projects/experiment/RobustnessTest/strong' ...
     ...
    ];

imdsVal = imageDatastore(valPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

inputSize = [227 227];
augVal = augmentedImageDatastore(inputSize, imdsVal);

load('run2_model.mat', 'trainedNet');

predLabels = classify(trainedNet, augVal);
trueLabels = imdsVal.Labels;

accuracy = mean(predLabels == trueLabels);
fprintf('\nRobustness Test Accuracy (Baseline Run2): %.2f%%\n', accuracy * 100);

fid = fopen('baseline_run2_robustness_strong_accuracy.txt', 'w');
fprintf(fid, 'Robustness Test Accuracy (Baseline Run2): %.2f%%\n', accuracy * 100);
fclose(fid);