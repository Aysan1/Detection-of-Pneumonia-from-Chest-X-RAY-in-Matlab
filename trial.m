load('net');

image = imread('n6.jpeg');
aug = augmentedImageDatastore([224 224],image, 'ColorPreprocessing', 'gray2rgb' );

% figure;
%imshow(aug);
[YPred,probs] = classify(net,aug)
%accuracy = mean(YPred == imdsTest.Labels)


