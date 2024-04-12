%% negative class
neg=dir('./CaltechFaces/my_train_non_face_scenes/*.jpg');

%% negative class augmentation
mkdir('./CaltechFaces/my2_train_non_face_scenes/')
outdir='./CaltechFaces/my2_train_non_face_scenes';

for ii=1:size(neg,1)
    im=imread([neg(ii).folder filesep neg(ii,1).name]);
    imwrite(im,[outdir filesep neg(ii,1).name]);
    
    [pp,ff,ee]=fileparts(neg(ii).name);
    im_flip=fliplr(im);
    imwrite(im_flip,[outdir filesep ff '_flip' ee]);

    im_ud = flipud(im); % Upside-down version
    imwrite(im_ud,[outdir filesep ff '_ud' ee]); % Save upside-down version
    
    for nrot=1:10 % Rotate the image multiple times
        imr = imrotate(im, 35*nrot, 'crop'); % Rotate by multiples of 35 degrees
        imwrite(imr,[outdir filesep ff '_r' num2str(nrot) ee]);
    end
end

%%
negativeFolder = './CaltechFaces/my2_train_non_face_scenes';
negativeImages = imageDatastore(negativeFolder);

%% positive class

faces = dir('./CaltechFaces/my_train_faces/*.jpg');
sz = [size(faces,1) 2];
varTypes = {'cell','cell'};
varNames = {'imageFilename','face'};
facesIMDB = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

for ii=1:size(faces,1)
    facesIMDB.imageFilename(ii) = {[faces(ii).folder filesep faces(ii).name]};
    facesIMDB.face(ii) = {[1 1 32 32]};
end

positiveInstances = facesIMDB;

%% VJ detector training

trainCascadeObjectDetector('myFaceDetector.xml',positiveInstances, ...
    negativeImages, 'FalseAlarmRate',0.05, 'NumCascadeStages',20);

%% visualize the results

detector = vision.CascadeObjectDetector('myFaceDetector.xml');
% detector = vision.CascadeObjectDetector();

imgs = dir('./CaltechFaces/test_scenes/test_jpg/*.jpg');

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector,img);
    
    detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'face');
    detectedImg = imresize(detectedImg,800/max(size(detectedImg)));
    
    figure(1),clf
    imshow(detectedImg)
    waitforbuttonpress
end

%% compute our results
load('./CaltechFaces/test_scenes/GT.mat');

detector = vision.CascadeObjectDetector('myFaceDetector.xml');
% detector = vision.CascadeObjectDetector();


imgs = dir('./CaltechFaces/test_scenes/test_jpg/*.jpg');

numImages = size(imgs,1);
results = table('Size', [numImages 2],...
    'VariableTypes',{'cell','cell'},...
    'VariableNames',{'face','Scores'});

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector,img);
    results.face{ii}=bbox;
    results.Scores{ii}=0.5+zeros(size(bbox,1),1);
end

% compute average precision
[ap, recall, precision]=evaluateDetectionPrecision(results,GT,0.2);
figure(2),clf
plot(recall,precision)
xlim([0 1])
ylim([0 1])
grid on
title(sprintf('Average Precision = %.1f',ap))















