function [gindex] = GSDI(Clabels,Dlabels,test_Clabels,test_Dlabels,hyp)

c = unique(Clabels);
d = unique(Dlabels);

D = zeros(length(c),length(d));
for i = 1:length(c)
   temp_Dlabel = Dlabels((Clabels==c(i)));
%    temp_Dlabel = [temp_Dlabel, test_Dlabels(test_Clabels==c(i))];
   for j = 1:length(d)
      D(i,d(j)) = length(find(temp_Dlabel==d(j))); %D stores the number of points in each class from each of the disjuncts, on the entire dataset
   end
end

a = -1*ones(length(c),length(d));
for i = 1:length(c)
   idx = find(test_Clabels==c(i));
   temp_hyp = hyp(idx);
   temp_Dlabel_test = test_Dlabels(idx);
   temp_corr = (temp_hyp==c(i));
   corr_disj=temp_Dlabel_test(temp_corr);
   for j = 1:length(d)
      if ~isempty(find(temp_Dlabel_test==d(j), 1))           
               a(i,d(j)) = length(find(corr_disj==d(j)))/length(find(temp_Dlabel_test==d(j)));
%                a(i,d(j)) = length(find((temp_corr)&(temp_Dlabel_test==d(j))))/length(find(temp_Dlabels_test==d(j)));
      end      
   end
end

AA=[min(0.1*length(Clabels),10) 1; 0.5*length(Clabels) 1];
BB=[0;log(10)];
%AA*XX=BB
XX=AA\BB;
alpha=XX(1);
beta=XX(2);
wt = exp(-(alpha*D + beta));
wt(wt>1)=1;

wt(D==0) = 0;
wt(a == -1) = 0;
a(a == -1) = 0;
% a
% wt
% a.*wt
% sum(a.*wt,2)
% sum(wt,2)
% (sum(a.*wt,2)./sum(wt,2))
gindex=prod(sum(a.*wt,2)./(sum(wt,2)+eps))^(1/length(c));
% gindex = gen_mean(sum(a.*wt,2)./sum(wt,2),-10)^(1/length(c));

end