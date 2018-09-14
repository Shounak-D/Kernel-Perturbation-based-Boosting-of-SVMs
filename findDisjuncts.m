function [Dlabels, optk, Fx_k] = findDisjuncts(data, labels, distanceMatrix)
% Identifying the disjuncts in the dataset

kvector = 1:floor(sqrt(length(labels)));
Dlabels=zeros(length(kvector),length(labels));
[n,~]=size(data);

indices = 1:n;
reducedDist = zeros(n,(n-1));
for i=1:n
    reducedDist(i,:) = distanceMatrix(i,indices~=i);
end
[~, sortedIndex]=sort(reducedDist, 2, 'ascend');
for i=1:n
    sortedIndex(i,sortedIndex(i,:)>=i) = sortedIndex(i,sortedIndex(i,:)>=i) + 1;
end
    
for k = 1:length(kvector)
    
    hTemp=hist(labels, unique(labels));
    korig = k;
    klimit=min(k,min(hTemp)); %truncating k so as to be less than or equal to the number of pts. in the smallest class
    gamma=sortedIndex(:,1:klimit); %finding the indices for the k nearest neighbours
    
    M=[];
    Dlabels_temp=zeros(1,length(labels));
    Q=[];
    cls=unique(labels);
    Delc = zeros(1,length(cls));
    del=1;
    for c=1:length(cls)
        del_c = 1;
        M=[];
        xc=find(labels==cls(c));
        while (~isempty(setdiff(xc,M)))
            delta=[];
            xm = setdiff(xc,M,'stable');
            xx=xm(1,1);
            Q = [Q xx];
            M=[M xx];
            delta=[delta xx];
            while ~isempty(Q)
                u = Q(1);
                Q(1) = [];
                kneighbours=(gamma(u,:));
                m=1;
                kn_cls=[];
                for i=1:length(kneighbours)
                    if(labels(kneighbours(i))==cls(c))
                        kn_cls(m)=kneighbours(i);
                        m=m+1;
                    end
                end
                v=setdiff(kn_cls,[M Q],'stable');
                M=[M v];
                delta=[delta v];
                Q = [Q v];
            end
            Dlabels_temp(delta)=del;
            del_c=del_c+1;
            del=del+1;
        end
        Delc(c) = del_c - 1;
    end
    Dlabels(korig,:) = Dlabels_temp; %storing the Dlabels for the current k
    Fx_k(korig)=sum(Delc); %storing the total number of disjuncts for the current k
    
end


[optk, optk_idx] = knee_pt(Fx_k,kvector,true);
if (abs(Fx_k(optk+1) - Fx_k(optk)) >= 0.33*abs(Fx_k(optk) - Fx_k(optk-1)))
    optk = optk + 1; 
    optk_idx = optk_idx + 1;
end

figure(1)
plot(kvector,Fx_k) %plotting the number of disjuncts against k
hold on;
plot(kvector,Fx_k,'b.')
plot(optk,Fx_k(optk_idx),'ro'); %pointing out the knee point
Dlabels=Dlabels(optk,:); %isolating the Dlabels for the knee point



fprintf('Finished identifying the disjuncts.\n');



end