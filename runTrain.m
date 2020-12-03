function runRetraining_2class(learnRT, maxEP, minBS)

warning off
addpath('./SOPPORTFILES')
MEANZERO = [103.939 116.779 123.68];
INTENSITYSCALE = 255;

% Import pre-trained model
% Download model2.mat from <https://uob-my.sharepoint.com/:u:/g/personal/eexna_bristol_ac_uk/EXH66HZ2rxlDrZBEoo5fgIABXYWYNcZl6N723jKesLdA9w?e=Zds50s>
% then save it in models/ folder
% -------------------------------------------------------------------------
netName = 'model2';
load(['models/',netName,'.mat']);

% data and output directories
% -------------------------------------------------------------------------
folderResult = 'results/';
dataDir = 'data/';
mkdir(folderResult)

% parameter
% -------------------------------------------------------------------------
hpatch = 227;
wpatch = 227;
hprange = -round(hpatch/2)+1:round(hpatch/2)-1;
wprange = -round(wpatch/2)+1:round(wpatch/2)-1;

% transfer to layers for training
% -------------------------------------------------------------------------
layers = netFineTune.Layers;

% Training
% -------------------------------------------------------------------------
tic
% pre-define 2 sets stored in dataDir/deform/*.png and dataDir/stratified/*.png
listimds = {dataDir};
imds = imageDatastore(listimds,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@readerAlex);

netName = ['Retrain_minBS',num2str(minBS),'_maxEp',num2str(maxEP),'_learnRT', num2str(learnRT)];

optionsTransfer = trainingOptions('sgdm',...
    'MiniBatchSize',minBS ,...
    'MaxEpochs',maxEP,...
    'InitialLearnRate',learnRT);%, ...
    %'CheckpointPath',folderResult);

[netFineTune, traininfo] = trainNetwork(imds,layers,optionsTransfer);
save([folderResult, netName,'.mat'],'netFineTune','traininfo');

% Validation Train
YTrain = classify(netFineTune,imds);
TTrain = imds.Labels;
accuracyTrain = sum(YTrain == TTrain)/numel(TTrain);

% Validation Test
% listimds = {'/mnt/storage/scratch/eexna/synthesised_patches/DSTvST_same_wrap_same/set2/'};
% imds = imageDatastore(listimds,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@readerAlex);
% YTrain = classify(netFineTune,imds);
% TTrain = imds.Labels;
% accuracyTest = sum(YTrain == TTrain)/numel(TTrain);
disp(['done train network:::: ',netName,' accuracyTrain=',num2str(accuracyTrain)]);%,' accuracyTest=',num2str(accuracyTest)])


