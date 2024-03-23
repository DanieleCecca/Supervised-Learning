accuracy_5x2=[];

for ndataset=1:4
    switch ndataset
        case 1, load dataset1.mat
        case 2, load dataset2.mat
        case 3, load dataset3.mat
        case 4, load dataset4.mat
        otherwise
    end
    
    accuracy_times=[];
    for ntimes=1:5
        % stratified sampling
        idx_tr=[];
        idx_te=[];
        for nclass=1:2
            u=find(labels==nclass);
            idx=randperm(numel(u));
            idx_tr=[idx_tr; u(idx(1:round(numel(idx)/2)))];
            idx_te=[idx_te; u(idx(1+round(numel(idx)/2):end))];
        end
        labels_tr=labels(idx_tr);
        labels_te=labels(idx_te);
        data_tr=data(idx_tr,:);
        data_te=data(idx_te,:);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % training classifier(s)
        % train on training split, test on test split
        SVM_LIN=fitcsvm(data_tr,labels_tr,'KernelFunction',...
            'linear','KernelScale',1);
        
        prediction=predict(SVM_LIN,data_te);
        accuracy1= numel(find(prediction==labels_te))/numel(labels_te);
        
        % reversing role of training and test:
        % train on test split, test on train split
        SVM_LIN=fitcsvm(data_te,labels_te,'KernelFunction',...
            'linear','KernelScale',1);
        
        prediction=predict(SVM_LIN,data_tr);
        accuracy2= numel(find(prediction==labels_tr))/numel(labels_tr);
        
        accuracy = (accuracy1+accuracy2)/2;
        accuracy_SVM_LIN(ntimes,1)=accuracy;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        SVM_RBF=fitcsvm(data_tr,labels_tr,'KernelFunction',...
             'gaussian','KernelScale',0.1); 

        prediction=predict(SVM_RBF,data_te);
        accuracy1= numel(find(prediction==labels_te))/numel(labels_te);
        
        % reversing role of training and test:
        % train on test split, test on train split
        SVM_RBF=fitcsvm(data_tr,labels_tr,'KernelFunction',...
             'gaussian','KernelScale',0.1); 
        
        prediction=predict(SVM_RBF,data_tr);
        accuracy2= numel(find(prediction==labels_tr))/numel(labels_tr);
        
        accuracy = (accuracy1+accuracy2)/2;
        accuracy_SVM_RBF(ntimes,1)=accuracy;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       KNN = fitcknn(data_tr,labels_tr,'Distance','Euclidean',...
          'NumNeighbors',10); 
        
        prediction=predict(KNN,data_te);
        accuracy1= numel(find(prediction==labels_te))/numel(labels_te);
        
        % reversing role of training and test:
        % train on test split, test on train split
       KNN = fitcknn(data_tr,labels_tr,'Distance','Euclidean',...
          'NumNeighbors',10);
        
        prediction=predict(KNN,data_tr);
        accuracy2= numel(find(prediction==labels_tr))/numel(labels_tr);
        
        accuracy = (accuracy1+accuracy2)/2;
        accuracy_KNN(ntimes,1)=accuracy;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        TREE = fitctree(data_tr,labels_tr,'SplitCriterion','gdi',...
            'MaxNumSplits', 10);
        
        prediction=predict(TREE,data_te);
        accuracy1= numel(find(prediction==labels_te))/numel(labels_te);
        
        % reversing role of training and test:
        % train on test split, test on train split
        TREE = fitctree(data_tr,labels_tr,'SplitCriterion','gdi',...
            'MaxNumSplits', 10);
        
        prediction=predict(TREE,data_tr);
        accuracy2= numel(find(prediction==labels_tr))/numel(labels_tr);
        
        accuracy = (accuracy1+accuracy2)/2;
        accuracy_TREE(ntimes,1)=accuracy;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end
    accuracy_5x2(ndataset,1)=mean(accuracy_SVM_LIN);
    accuracy_5x2(ndataset,2)=mean(accuracy_SVM_RBF);
    accuracy_5x2(ndataset,3)=mean(accuracy_KNN);
    accuracy_5x2(ndataset,4)=mean(accuracy_TREE);
end

disp(accuracy_5x2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the ranks
for i = 1:size(accuracy_5x2, 1)
    [~, sortIdx] = sort(accuracy_5x2(i, :), 'descend');
    ranks(i, sortIdx) = 1:numel(sortIdx);
end

display(ranks)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Calculate the mean ranks for each classifier across all datasets
meanRanks = mean(ranks, 1);
meanRanks


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Frieman test
%if we pass 1- accuracy we obtain the same rank order that we obtained
%above
[p,tbl,stats] = friedman(accuracy_5x2,1,'off');

%p-value
p
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the critical difference (CD) value
num_models = size(accuracy_5x2, 1);
num_datasets = size(accuracy_5x2, 2);
q_alpha = 2.569 %alpha 0.05
CD = q_alpha * sqrt((num_models * (num_models + 1)) / (6 * num_datasets));
display(CD)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[c,m,h,gnames] = multcompare(stats)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


