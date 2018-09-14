function kval = rbf_kernelDist1(distanceMatrix,rbf_sigma,dotProduct)
%RBF_KERNEL Radial basis function kernel for SVM functions when distance
%matrix and/or dot product matrix are given

if rbf_sigma > 0
    kval = exp(-(1/(2*rbf_sigma^2))*distanceMatrix);
elseif rbf_sigma==-1
    kval = dotProduct;
else
    kval = (dotProduct + 1).^( -1 * rbf_sigma );
end

if issparse(kval)
    kval = full(kval);
end


end

