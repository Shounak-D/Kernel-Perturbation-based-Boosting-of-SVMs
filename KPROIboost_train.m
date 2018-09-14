function [ktr,alpha,alpha1,alpha2,fxtrain,lambda,lambdainit,bias,biasinit,Dtr] = KPROIboost_train(train_x,train_y,param,step,T,c,sigma,distTr)

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
[h,fxtrain,lambdainit,biasinit,~,~]=SVM(train_x,train_y,c,sigma,K);
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

%% calculating param x (avg. 1-NN distance) for the 2 classes
%all points
distt_all = distTr;

%majority pts
distt_maj_pts = distTr(find(train_y==-1),find(train_y==-1));
req_distmaj = zeros(sum(train_y==-1),1);
for i = 1:(sum(train_y==-1))
    req_distmaj(i)=min(distt_maj_pts(i,(distt_maj_pts(i,:)~=0))); 
end
distance_metric_maj = param*mean(req_distmaj); %radius of influence for the majority class
% distance_metric_maj= param*min(req_distmaj);
% distance_metric_maj= param*max(req_distmaj);

%minority pts
distt_min_pts = distTr(find(train_y==1),find(train_y==1));
req_distmin = zeros(sum(train_y==1),1);
for i = 1:(sum(train_y==1))
    req_distmin(i)=min(distt_min_pts(i,(distt_min_pts(i,:)~=0))); 
end
distance_metric_min = param*mean(req_distmin); %radius of influence for the minority class
% distance_metric_min= param*min(req_distmin);
% distance_metric_min= param*max(req_distmin);

%% The KP-ROI Loop
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
%     gerror=g_mean_performance(h,train_y)/100;
    eror(t)=sqrt((1-tpr(t))^2+(1-tnr(t))^2);
    alpha(t)=0.5*log((sqrt(2)-eror(t))/(eror(t)+eps));
    
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
    
    %% warping the ktr value of the required datapoints for the 't'th iteration
    correct=find(h==train_y');
    incorrect=find(h~=train_y');

    inc_maj=incorrect(train_y(incorrect)==1);
    inc_min=incorrect(train_y(incorrect)==-1);
    
    for i=1:length(inc_maj)
        idx_inc_maj = find(distt_all(inc_maj(i),:)<=distance_metric_maj);
        for j=1:length(idx_inc_maj)
            correct(correct==idx_inc_maj(j))=[];
        end
    end
    for i=1:length(inc_min)
        idx_inc_min = find(distt_all(inc_min(i),:)<=distance_metric_min);
        for j=1:length(idx_inc_min)
            correct(correct==idx_inc_min(j))=[];
        end
    end
        
    if t~=T
        ktr((t+1),:)=ktr(t,:);
        for i=1:length(correct)
               ktr((t+1),correct(i))=ktr(t,correct(i))+step;
        end
           
    end

end

% plot(1:T,eror)
% figure
% plot([1:T],tpr)
% figure
% plot([1:T],tnr)
non_index_trnc = (eror<=1/sqrt(2)) & (eror<=error_init) & (tpr>=tpr_init);
if(sum(non_index_trnc)==0)
    non_index_trnc = (eror<=1/sqrt(2)) & (tpr>=tpr_init);
end
alpha(~non_index_trnc) = 0;
alpha1(alpha1<0) = 0;
alpha2(alpha2<0) = 0;

end
 