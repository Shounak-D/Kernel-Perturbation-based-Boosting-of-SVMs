function [ftpr,ftnr,prec,gm,yPred,ftpr1,ftnr1,prec1,gm1,yPred1,ftpr2,ftnr2,prec2,gm2,yPred2] = ...
          KPROIboost_testKtr(train_x,train_y,test_x,test_y,Dtr,ktr,sigma,lambda,lambdainit,bias,biasinit,alpha,alpha1,alpha2,distTrTs)

T=size(Dtr,1);
k=1; %the number of neighbours to be used for ktr estimation

if (nargin<13)
    distTrTs = repmat(sqrt(sum(train_x.^2,2).^2),1,size(test_x,1)) - 2*(train_x*test_x') + repmat(sqrt(sum(test_x.^2,2)'.^2),size(train_x,1),1);
    K_tr_ts=rbf_kernelDist1(distTrTs,sigma);
else
    K_tr_ts=rbf_kernelDist1(distTrTs,sigma);
end
[~,fxtestinit]=SVM_classify(test_x,train_x,train_y,lambdainit,biasinit,sigma,K_tr_ts);
% posMask = (test_y==1); negMask = ~posMask;
% out_posMask = (test_label1==1); out_negMask = ~out_posMask;
% tp = sum(posMask.*out_posMask');
% fp = sum(negMask.*out_posMask');
% tn = sum(negMask.*out_negMask');
% fn = sum(posMask.*out_negMask');
% precision = tp/(tp + fp);
% fintpr = tp/(tp + fn);
% fintnr = tn/(tn + fp);
% gmeansinit = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) );
% fmeasureinit=2*precision*fintpr/(precision+fintpr);
% aucinit= (fintpr+fintnr)/2;
% fprintf('The initial testing G-means is: %d\n',gmeansinit)
% fprintf('The initial testing fmeasure is: %d\n',fmeasureinit)
% fprintf('The initial testing auc is: %d\n',aucinit)

distTsTr = distTrTs';
[~, sortedIndex]=sort(distTsTr, 2, 'ascend');
gamma=sortedIndex(:,1:k);

%% ktr case
for t=1:T
    
    if k==1
        kts(t,:) = ktr(t,gamma);
    else
        for i = 1:length(test_y)
            kts(t,i) = mean(ktr(t,gamma(i,:)));
        end
    end
    
    if(t==1)
        Dts(t,:)=exp(-1.*kts(t,:).*(fxtestinit.^2));
    else
        Dts(t,:)=exp(-1.*kts(t,:).*(fx_test(t-1,:).^2)); 
    end
    if(t==1)
        Knew{t} = (Dtr(t,:)'*Dts(t,:)).*K_tr_ts;
    else
        Knew{t} = (Dtr(t,:)'*Dts(t,:)).*Knew{t-1};
    end
    
    [hyp_label(t,:),fx_test(t,:)]=SVM_classify(test_x,train_x,train_y,lambda(:,t),bias(:,t),sigma,Knew{t});
        
end

fxtest=sum((repmat(alpha',1,size(test_x,1)).*hyp_label),1);
yPred = sign(fxtest);
posMask = (test_y==1); negMask = ~posMask;
out_posMask = (yPred==1); out_negMask = ~out_posMask;
tp = sum(posMask.*out_posMask');
fp = sum(negMask.*out_posMask');
tn = sum(negMask.*out_negMask');
fn = sum(posMask.*out_negMask');
prec = tp/(tp + fp + eps);
ftpr = tp/(tp + fn + eps);
ftnr = tn/(tn + fp + eps);
gm = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) ); %Final Gmeans using alpha
% fprintf('The final testing G-means is: %d\n',gm);

fxtest1=sum((repmat(alpha1',1,size(test_x,1)).*hyp_label),1);
yPred1 = sign(fxtest1);
posMask = (test_y==1); negMask = ~posMask;
out_posMask = (yPred1==1); out_negMask = ~out_posMask;
tp = sum(posMask.*out_posMask');
fp = sum(negMask.*out_posMask');
tn = sum(negMask.*out_negMask');
fn = sum(posMask.*out_negMask');
prec1 = tp/(tp + fp + eps);
ftpr1 = tp/(tp + fn + eps);
ftnr1 = tn/(tn + fp + eps);
gm1 = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) ); %Final Gmeans using alpha1
% fprintf('The final testing G-means is: %d\n',gm);

fxtest2=sum((repmat(alpha2',1,size(test_x,1)).*hyp_label),1);
yPred2 = sign(fxtest2);
posMask = (test_y==1); negMask = ~posMask;
out_posMask = (yPred2==1); out_negMask = ~out_posMask;
tp = sum(posMask.*out_posMask');
fp = sum(negMask.*out_posMask');
tn = sum(negMask.*out_negMask');
fn = sum(posMask.*out_negMask');
prec2 = tp/(tp + fp + eps);
ftpr2 = tp/(tp + fn + eps);
ftnr2 = tn/(tn + fp + eps);
gm2 = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) ); %Final Gmeans using alpha2
% fprintf('The final testing G-means is: %d\n',gm);

end
