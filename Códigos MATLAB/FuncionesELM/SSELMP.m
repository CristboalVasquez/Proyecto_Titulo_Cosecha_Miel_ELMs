function elmModel=sselm(Xl,Yl,Xu,L,paras)

[l,elmModel.InputDim]=size(Xl);
u=size(Xu,1);
N=l+u;

elmModel.InputWeight=rand(elmModel.InputDim,paras.NumHiddenNeuron)*2-1;

elmModel.Kernel=paras.Kernel;
switch paras.Kernel
    case 'sigmoid'
        H=1 ./ (1 + exp(-[Xl;Xu]*elmModel.InputWeight));
    case 'rbf'
        
end
Hl=H(1:l,:);
clear Xl Xu

Y=[Yl;zeros(u,size(Yl,2))];
Cl_diag=ones(l,1);

Cl=diag(paras.C*Cl_diag);
if  (paras.NumHiddenNeuron>N) 
    C=diag([paras.C*Cl_diag;zeros(u,1)]); 
end

t_elm_start=tic;
if  (paras.NumHiddenNeuron<N)
    elmModel.OutputWeight=(eye(paras.NumHiddenNeuron)+Hl'*Cl*Hl+paras.lambda*H'*L*H)\ (Hl'*Cl* Yl); %formula uitilizada
else
    A=eye(N)+(C+paras.lambda*L)*(H*H');
    B=C*Y;
    D=A\B;
    elmModel.OutputWeight=H'*D;
    elmModel.OutputWeight0=H'*((eye(N)+(C+paras.lambda*L)*(H*H'))\(C*Y));
    norm(elmModel.OutputWeight-elmModel.OutputWeight0)
end
elmModel.TrainTime=toc(t_elm_start);

out_train=Hl*elmModel.OutputWeight;
elmModel.TrainRMSE=sqrt(mean((Yl-out_train).^2));

if ~paras.NoDisplay
    disp(['Traning time is ',num2str(elmModel.TrainTime)])
    disp(['Traning RMSE is ',num2str(elmModel.TrainRMSE)])
end