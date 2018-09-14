%% Implementation of KPBoost-SVM

% Required pre-loaded data in the workspace
%-------------------------------------------
% data -> dataset, rows correspond to data points
% labels -> class labels to be used for clustering validity (Minority class 
%           should have label +1 and the majority class should have label -1)

%% Initialization
c = input('Enter the Cost of Regularization: ');
sigma = input('Enter the kernel width for the initial RBF kernel: '); 
stepSize = input('Enter the step for kernel perturbation: '); 
param = input('Enter the factor determining the Region of Influence: '); %should belong to the set {0.6, 0.7, 0.8}
T_boost=10;
kpart = 5;

%% Calculating the Distance and Dot Product Matrices.
[n,~]=size(data);
% distanceMatrix=zeros(n,n);
% for i=1:n
%     distanceMatrix(:,i) = sum((repmat(data(i,:),n,1) - data).^2,2);
% end
% distanceMatrix = distanceMatrix';
dotProduct = data*data';
distanceMatrix = repmat(sqrt(sum(data.^2,2).^2),1,n) - 2*dotProduct + repmat(sqrt(sum(data.^2,2)'.^2),n,1);
fprintf('Finished calculating the Distance and Dot Product Matrices.\n');

%% Finding the Disjuncts in the dataset
[Dlabels, optk, Fx_k] = findDisjuncts(data, labels, distanceMatrix);

%% Creating the partitions for 5-fold cross-validation (should not be applied to any dataset having less than 5 pts. in any of the classes)
k = kpart; %number of partitions to be made
uniq=unique(labels);
cls_num = length(uniq);
sIze = zeros(1,cls_num);
leftover = zeros(1,cls_num);
for j = 1:cls_num
    sIze(j) = floor(size(data(labels==uniq(j),:),1)/k);
    leftover(j) = size(data(labels==uniq(j),:),1) - (k * floor(size(data(labels==uniq(j),:),1)/k));
end
sIze = repmat(sIze,k,1);
flag = 0;
for j = 1:cls_num
    if flag==0
        sIze_idx = 1;
    else
        sIze_idx = k;
    end
    while leftover(j) > 0
        sIze(sIze_idx,j) = sIze(sIze_idx,j) + 1;
        leftover(j) = leftover(j) - 1;
        if flag==0
            sIze_idx = sIze_idx + 1;
        else
            sIze_idx = sIze_idx - 1;
        end
    end
    flag = ~flag;
end
sIze_cum = cumsum(sIze);
sIze_cum = circshift(sIze_cum,1);

rand_IDX = randperm(length(labels));
xx = data(rand_IDX,:);
yy = labels(rand_IDX);
Dlabels1 = Dlabels(rand_IDX)';

X_train = cell(k,1);
Y_train = cell(k,1);
train_Dlabels = cell(k,1);
X_test = cell(k,1);
Y_test = cell(k,1);
test_Dlabels = cell(k,1);
part_idx = cell(k,2);

for i = 1:k
    
    % extracting the training and test sets for a particular partition
    x_test = []; y_test = []; x_train = []; y_train = []; D_test=[]; D_train=[]; part_idx_test = []; part_idx_train = [];
    for j = 1:cls_num
        temp_idx = rand_IDX(yy==uniq(j));
        temp_idx = circshift(temp_idx,[0 -1*sIze_cum(i,j)]);
        part_idx_test = [part_idx_test temp_idx(1:sIze(i,j))];
        part_idx_train = [part_idx_train temp_idx((sIze(i,j)+1):end)];
        x_temp = xx(yy==uniq(j),:);
        x_temp = circshift(x_temp,-1*sIze_cum(i,j));
        D_temp = Dlabels1(yy==uniq(j));
        D_temp = circshift(D_temp,-1*sIze_cum(i,j));
        y_temp = uniq(j) * ones(size(x_temp,1),1);
        x_test = [x_test; x_temp(1:sIze(i,j),:)];
        D_test = [D_test; D_temp(1:sIze(i,j))];
        y_test = [y_test; y_temp(1:sIze(i,j))];
        x_train = [x_train; x_temp((sIze(i,j)+1):end,:)];
        D_train = [D_train; D_temp((sIze(i,j)+1):end)];
        y_train = [y_train; y_temp((sIze(i,j)+1):end)];
    end
    
    %storing x_train, y_train, x_test, y_test, D_train, and D_test here for later use
    part_idx{i,1} = part_idx_train;
    part_idx{i,2} = part_idx_test;
    X_train{i}=x_train;
    Y_train{i}=y_train;
    train_Dlabels{i}=D_train;
    X_test{i}=x_test;
    Y_test{i}=y_test;
    test_Dlabels{i}=D_test;
    
end
fprintf('Finished creating partitions for %d-fold cross-validation.\n',k);
clearvars -except data labels Dlabels optk Fx_k k X_train Y_train train_Dlabels X_test Y_test test_Dlabels rand_IDX part_idx distanceMatrix dotProduct...
                  sortedIndex c sigma stepSize param T_boost

%% Looping over the partitions
for i = 1:k
    train_x=X_train{i};
    train_y=Y_train{i};
    test_x=X_test{i};
    test_y=Y_test{i};
    testDlabels=test_Dlabels{i};
    distTr = distanceMatrix(part_idx{i,1},part_idx{i,1});
    distTrTs = distanceMatrix(part_idx{i,1},part_idx{i,2});
    
    %run kp
    [alpha,alpha1,alpha2,~,lambda,bias,Dtr] = KPboost_train(train_x,train_y,stepSize,T_boost,c,sigma,distTr);
    [tpr_kp(i),tnr_kp(i),prec_kp(i),~,yPred] = KPboost_test(train_x,train_y,test_x,test_y,Dtr,sigma,lambda,bias,alpha,alpha1,alpha2,distTrTs);
    [gindexKp(i)] = GSDI(labels,Dlabels,test_y,testDlabels,yPred);
    fprintf('Finished running KPBoost-SVM for partition %d.\n',i);
    
    %run kproi
    [ktr,alpha,alpha1,alpha2,~,lambda,lambdainit,bias,biasinit,Dtr] = KPROIboost_train(train_x,train_y,param,stepSize,T_boost,c,sigma,distTr);
    [tpr_ktr(i),tnr_ktr(i),prec_ktr(i),~,yPred] = KPROIboost_testKtr(train_x,train_y,test_x,test_y,Dtr,ktr,sigma,lambda,lambdainit,bias,biasinit,alpha,alpha1,alpha2,distTrTs);
    [gindexKproi_ktr(i)] = GSDI(labels,Dlabels,test_y,testDlabels,yPred);
    fprintf('Finished running KPBoostROI-SVM for partition %d.\n',i);
        
end
clearvars i j train_x train_y test_x test_y distTr distTrTs

%finding average (over the k-FCV partitions) performance for the proposed methods (KPBoost-SVM and KPBoostROI-SVM)
Kp_tpr = mean(tpr_kp,1);
Kproi_ktr_tpr = mean(tpr_ktr,1);
Kp_tnr = mean(tnr_kp,1);
Kproi_ktr_tnr = mean(tnr_ktr,1);
Kp_prec = mean(prec_kp,1);
Kproi_ktr_prec = mean(gindexKp,1);
gmean_kp = sqrt(trp_kp.*tnr_kp);
Kp_gmean = mean(gmean_kp,1);
gmean_ktr = sqrt(trp_ktr.*tnr_ktr);
Kproi_ktr_gmean = mean(gmean_ktr,1);
Kp_gsdi = mean(prec_kp,1);
Kproi_ktr_gsdi = mean(gindexKproi_ktr,1);
