imds = imageDatastore('chData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[imdsTrain,imdsTest] = splitEachLabel(imds, 0.70,'randomize');

augimdsTrain  = augmentedImageDatastore([224 224], imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsTest  = augmentedImageDatastore([224 224], imdsTest, 'ColorPreprocessing', 'gray2rgb');

%specify the layers in the network
layers = [
    imageInputLayer([224 224 3]) 
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2) 
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer 
    maxPooling2dLayer(2, 'stride', 2)
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsTest, ...
    'Verbose',false, ...
    'Plots','training-progress');




net = trainNetwork(augimdsTrain,layers,options);

[YPred,probs] = classify(net,augimdsTest);
accuracy = mean(YPred == imdsTest.Labels);



idx = randperm(numel(augimdsTest.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end






































