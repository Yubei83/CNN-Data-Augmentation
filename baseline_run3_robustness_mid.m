clc; clear; close all;

valPath = ['E:/study/MATH 3001 Projects/projects/experiment/RobustnessTest/mid' ...
     ...
    ];

imdsVal = imageDatastore(valPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

inputSize = [227 227];
augVal = augmentedImageDatastore(inputSize, imdsVal);

load('run3_model.mat', 'trainedNet');

predLabels = classify(trainedNet, augVal);
trueLabels = imdsVal.Labels;

accuracy = mean(predLabels == trueLabels);
fprintf('\nRobustness Test Accuracy (Baseline Run3): %.2f%%\n', accuracy * 100);

fid = fopen('baseline_run3_robustness_mid_accuracy.txt', 'w');
fprintf(fid, 'Robustness Test Accuracy (Baseline Run3): %.2f%%\n', accuracy * 100);
fclose(fid);