function [trainhyp,fx,alphas,offset,svIndex,testK] = SVM(X,Y,C,rbf_sigma,testK) 

Xorign=X;
Yorigin=Y;
% if isempty(C)
%     C = input('Enter the cost of Regularization: ');
% end
% if isempty(rbf_sigma)
%     display('Positive kernel parameter runs the RBF kernel, negative parameter runs the Linear kernel...');
%     rbf_sigma = input('Enter the value of the kernel parameter: ');
% end
% smoOptions = svmsmoset;
smoOptions.TolKKT = 0.001;
smoOptions.MaxIter = 50000; %25000;
smoOptions.KKTViolationLevel = 0.1; %0.05;
boxconstraint = C*ones(length(Y), 1); %include to run C-SVM
P = ones(length(Yorigin), 1); %include to run C-SVM
if(nargin<5)
    testK = rbf_kernel21(Xorign, Xorign,rbf_sigma);
end
[alphas, offset] = Kern_SMO(X, Y, boxconstraint, P, testK, smoOptions);
svIndex = find((alphas>(eps))&(alphas<(C-eps)));
%     testK = rbf_kernel2(Xorign, Xorign);   %gaussian rbf
%     testK = (Xorign(svIndex,:)*Xorign');
    
fx = ((alphas.*Y)'*testK + offset);
    
trainhyp = sign(fx);
    
end
 