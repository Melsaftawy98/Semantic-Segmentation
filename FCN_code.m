%% 1− Setup vgg network
vgg16 ( ) ;

%% 2−Label Data ( i . e the o b tai n e d data downloaded from Matlab so
t h e r e ’ s no need to do i t )

%% 3− C rea te a d a t a s t o r e f o r o r i g i n a l images and l a b e l e d images
imds = imageDatastore(”C:\Users\Lenovo\Documents\MATLAB\Examples\R2021a\deep_learning_shared\
TrainAndDeployFullyConvolutionalNetworksExample\images\701_Stills_Raw_full”);

pxds = imageDatastore (”C:\Users\Lenovo\Documents\MATLAB\Examples\R2021a\deep_learning_shared\
TrainAndDeployFullyConvolutionalNetworksExample\labels”);

disp(imds);
I = readimage(imds , 1);
histeq(I);
imshow(I);

%% 4− Load camVid pixel−labeled images
 classes =[
 ”Sky”
 ”B uil di ng ”
 ”Pole ”
 ”Road”
 ”Pavement”
 ”Tree ”
 ”SignSymbol ”
 ”Fence ”
 ”Car”
 ” Pedestrian ”
 ” Bicyclist ”
 ];
 
labelIDs = camvidPixelLabelIDs();
labelDir = fullfile("labels");

%% 5- Create a pixel label datastore
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
C = readimage(pxds,1);
cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
imshow(B)
pixelLabelColorbar(cmap,classes);

%% 6- Analyzed datasheet statistics
tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%% 7- Resize the dataset
imageFolder= fullfile('TrainAndDeployFullyConvolutionalNetworksExample\ResizedImages','imagesResized',filesep);
imds = resizeCamVidImages(imds,imageFolder);
labelFolder = fullfile('TrainAndDeployFullyConvolutionalNetworksExample\ResizedImages','labelsResized',filesep);
pxds = resizeCamVidPixelLabels(pxds, labelFolder);

%% 8- Prepare training and testing sets (70% Train / 15% Validation / 15% Test)
[imdsTrain,imdsTest,pidsTrain,pidsTest] = partitionCamVidData(imds,pxds);
numTrainingImages = numel(imdsTrain.Files);
numTestingImages = numel(imdsTest.Files);

%% 9- Create the Network
imageSize = [360 480 3];
numClasses = numel(classes);
Igraph = segnetLayers(imageSize,numClasses,'vgg16');

%% 10- Balance classes using class weight

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name, 'ClassWeights', classWeights);
Igraph = removeLayers(Igraph,'pixelLabels');
Igraph = addLayers(Igraph, pxLayer);
Igraph = connectLayers(Igraph,'softmax','labels');


%% 11-Data Augmentation

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10], 'RandYTranslation', [-10 10]);
pixmids = pixelLabelImageDatastore(imdsTrain,pidsTrain,...
    'DataAugmentation',augmenter);

%% 12-Setting training Options

options = trainingOptions('sgdm',...
    'Momentum',0.9,...
    'InitialLearnRate',1e-3,...
    'L2Regularization', 0.0005,...
    'MaxEpochs',2,...
    'MiniBatchSize',1,...
    'Shuffle','every-epoch',...
    'VerboseFrequency',2);

%% 13-Start Training

[net, info] = trainNetwork(pixmids,Igraph,options);
save net net;