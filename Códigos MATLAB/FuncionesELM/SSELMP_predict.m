function [RMSE,R,predict]=sselm_predict(X,Y,elmModel)

switch elmModel.Kernel
    case 'sigmoid'
        H=1 ./ (1 + exp(-X*elmModel.InputWeight));
    case 'rbf'
        
end

predict=H*elmModel.OutputWeight;

MSE=mean((Y-predict).^2);
RMSE=sqrt(MSE);

if std(Y)>0 && std(predict)>0
    mat_corr = corrcoef(Y, predict);
    R = mat_corr(1,2);
else
    R = NaN;
end