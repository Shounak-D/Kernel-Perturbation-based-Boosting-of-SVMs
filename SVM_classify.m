function [test_label,fxtest,Knew] = SVM_classify(test,train,train_y,lambda,bias,sigma,Knew)
% if (size(test,1)~=size(Knew,2) || size(train,1) ~= size(Knew,1))
%     error('Improper Kernel');
% else
    if(nargin<7)
        Knew=rbf_kernel21(train,test,sigma); %calculating kernel
        fxtest=((lambda.*train_y)'*Knew + bias);
        test_label = sign((lambda.*train_y)'*Knew + bias);
    else
        fxtest=((lambda.*train_y)'*Knew + bias);
        test_label = sign((lambda.*train_y)'*Knew + bias);
    end
% end

end

