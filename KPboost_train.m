function [alpha,alpha1,alpha2,fxtrain,lambda,bias,Dtr] = KPboost_train(train_x,train_y,step,T,c,sigma,distTr)

n=length(train_y);
% step=input('enter step size : ');
% T=input('enter number or iterations : ');
ktr=zeros(T,n); 
Dtr=zeros(T,n);
alpha=zeros(1,T); %the alpha denoting weights of the iterations (based on iteration performance only)
alpha1 = zeros(1,T); %the alpha corresponding to minimization of sum error
alpha2 = zeros(1,T); %the alpha corresponding to minimization of sum of squares error
D = (1/n)*ones(n,1); %weightage for individual datapoints due to boosting
% fxtrain=zeros(T,n);
% lambda=[];bias=[];

K = rbf_kernelDist1(distTr,sigma);
[h,fxtrain]=SVM(train_x,train_y,c,sigma,K);
posMask = (train_y==1); negMask = ~posMask;
out_posMask = (h==1); out_negMask = ~out_posMask;
tp = sum(posMask.*out_posMask');
fp = sum(negMask.*out_posMask');
tn = sum(negMask.*out_negMask');
fn = sum(posMask.*out_negMask');
% precision = tp/(tp + fp);
tpr_init = tp/(tp + fn);
tnr_init = tn/(tn + fp);
error_init=sqrt((1-tpr_init)^2+(1-tnr_init)^2);

% gmeansValinit = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) ); %Initial Validation Gmeans
% fprintf('\nThe initial validation G-means is: %d\n',gmeansValinit);

h=h(:)';
fxtrain_init=fxtrain(:)';
fxtrain=fxtrain(:)';
for t=1:T
    
    if(t==1)
        Dtr(t,:) = exp(-1.*ktr(t,:).*(fxtrain_init.^2));
    else
        Dtr(t,:) = exp(-1.*ktr(t,:).*(fxtrain(t-1,:).^2));
    end
    K=((Dtr(t,:))'*Dtr(t,:)).*K;
    [h,fxtrain(t,:),lambda(:,t),bias(t),~,~]=SVM(train_x,train_y,c,sigma,K);
    posMask = (train_y==1); negMask = ~posMask;
    out_posMask = (h==1); out_negMask = ~out_posMask;
    tp = sum(posMask.*out_posMask');
    fp = sum(negMask.*out_posMask');
    tn = sum(negMask.*out_negMask');
    fn = sum(posMask.*out_negMask');
    tpr(t) = tp/(tp + fn);
    tnr(t) = tn/(tn + fp);
%     gerror(t) = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) );
    eror(t) = sqrt((1-tpr(t))^2+(1-tnr(t))^2);
    alpha(t) = 0.5*log((sqrt(2)-eror(t))/(eror(t)+eps));
    
    tpD = sum(posMask.*out_posMask'.*D);
    fpD = sum(negMask.*out_posMask'.*D);
    tnD = sum(negMask.*out_negMask'.*D);
    fnD = sum(posMask.*out_negMask'.*D);
%     alpha1(t) = 0.5*log(((tpD/(tp+fn)) + (tnD/(tn+fp)))/((fnD/(tp+fn)) + (fpD/(tn+fp))));
    alpha1(t) = 0.5*log(((tpD/(tpD+fnD)) + (tnD/(tnD+fpD)))/((fnD/(tpD+fnD)) + (fpD/(tnD+fpD)) + eps));
    alpha2(t) = 0.25*log(((tpD/(tp+fn))^2 + (tnD/(tn+fp))^2)/((fnD/(tp+fn))^2 + (fpD/(tn+fp))^2) + eps);
%     fprintf('iteration %d: gmeans= %d , alpha= %d \n',t,gerror,alpha(t));

    %update weights
    D = D.*exp(-1.*alpha(t).*train_y.*fxtrain(t,:)');
    D = D./sum(D);

%% identifying the points in need of resolution preservation
    correct=find(h==train_y');
%     display(ee);
    if t~=T
        ktr((t+1),:)=ktr(t,:);
        for i=1:length(correct)
               ktr((t+1),correct(i))=ktr(t,correct(i))+step;
        end
           
    end
    
end

% plot([1:T],eror)
non_index_trnc = (eror<=1/sqrt(2)) & (eror<=error_init) & (tpr>=tpr_init);
if(sum(non_index_trnc)==0)
    non_index_trnc = (eror<=1/sqrt(2)) & (tpr>=tpr_init);
end
alpha(~non_index_trnc) = 0;
alpha1(alpha1<0) = 0;
alpha2(alpha2<0) = 0;

end
 