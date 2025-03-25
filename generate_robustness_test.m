clc; clear; close all;

srcPath = 'E:/study/MATH 3001 Projects/projects/experiment/Train';
dstPath = 'E:/study/MATH 3001 Projects/projects/experiment/RobustnessTest/strong';
inputSize = [227 227];

imds = imageDatastore(srcPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsSubset, ~] = splitEachLabel(imds, 100, 'randomized');

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

    img_aug = imgaussfilt(img, 3.5);                            
    img_aug = imrotate(img_aug, randi([-25, 25]), 'crop');     
    img_aug = imadjust(img_aug, stretchlim(img_aug), [], 0.4); 
    img_aug = imnoise(img_aug, 'salt & pepper', 0.03);         

    imwrite(img_aug, fullfile(outDir, [fileName, ext]));

    if mod(i, 200) == 0
        fprintf('Processed %d / %d images...\n', i, numel(imdsSubset.Files));
    end
end

fprintf('\n High-perturbation set generated at: %s\n', dstPath);
