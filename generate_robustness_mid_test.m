
clc; clear; close all;

srcPath = 'E:/study/MATH 3001 Projects/projects/experiment/Train';
dstPath = 'E:/study/MATH 3001 Projects/projects/experiment/RobustnessTest/mid';
inputSize = [227 227];

imds = imageDatastore(srcPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsSubset = splitEachLabel(imds, 100, 'randomized');
reset(imdsSubset);

for i = 1:numel(imdsSubset.Files)
    img = readimage(imdsSubset, i);
    label = char(imdsSubset.Labels(i));
    [~, fileName, ext] = fileparts(imdsSubset.Files{i});

    outDir = fullfile(dstPath, label);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    img = imresize(img, inputSize);
    img_aug = imgaussfilt(img, 1.0);                         
    img_aug = imrotate(img_aug, randi([-10, 10]), 'crop');  
    img_aug = imadjust(img_aug, [], [], 0.9);                

    imwrite(img_aug, fullfile(outDir, [fileName, ext]));

    if mod(i, 200) == 0
        fprintf('Processed %d / %d images...\n', i, numel(imdsSubset.Files));
    end
end

fprintf('\n Moderate robustness test set saved to: %s\n', dstPath);
