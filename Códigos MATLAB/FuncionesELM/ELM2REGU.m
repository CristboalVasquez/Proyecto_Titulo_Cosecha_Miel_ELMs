function [TrainingTime_r, TestingTime_r, TrainingAccuracy_r, TestingAccuracy_r, ...
    trainR_r, testR_r, trainMSE_r, testMSE_r, trainR2_r, testR2_r, TY_R, Y, TV_T, T] = ...
    ELM2REGU(T, P, TT, PP, Elm_Type, NumberofHiddenNeurons, ActivationFunction, C)
% ELM2REGU - Extreme Learning Machine con Regularización
%
% Input:
%   T, P              - Salidas y entradas de entrenamiento (formato: filas=features)
%   TT, PP            - Salidas y entradas de testeo
%   Elm_Type          - 0 para regresión; 1 para clasificación
%   NumberofHiddenNeurons - Número de neuronas ocultas
%   ActivationFunction    - 'sig', 'sin', 'hardlim', 'tribas', 'radbas'
%   C                 - Parámetro de regularización
%
% Output:
%   TrainingTime_r    - Tiempo de entrenamiento (s)
%   TestingTime_r     - Tiempo de testeo (s)
%   TrainingAccuracy_r - RMSE de entrenamiento
%   TestingAccuracy_r  - RMSE de testeo
%   trainR_r          - Correlación R de entrenamiento
%   testR_r           - Correlación R de testeo
%   trainMSE_r        - MSE de entrenamiento
%   testMSE_r         - MSE de testeo
%   trainR2_r         - R² de entrenamiento
%   testR2_r          - R² de testeo
%   TY_R              - Predicciones de testeo
%   Y                 - Predicciones de entrenamiento
%   TV_T              - Valores reales de testeo
%   T                 - Valores reales de entrenamiento

%%%%%%%%%%% Macro definition
REGRESSION = 0;
CLASSIFIER = 1;

%%%%%%%%%%% Inicializar variables de salida
trainR_r = NaN;
testR_r = NaN;
trainMSE_r = NaN;
testMSE_r = NaN;
trainR2_r = NaN;
testR2_r = NaN;

%%%%%%%%%%% Load datasets
TV.T = TT;
TV.P = PP;

NumberofTrainingData = size(P, 2);
NumberofTestingData = size(TV.P, 2);
NumberofInputNeurons = size(P, 1);

%%%%%%%%%%% Preprocessing para clasificación
if Elm_Type ~= REGRESSION
    sorted_target = sort(cat(2, T, TV.T), 2);
    label = zeros(1, 1);
    label(1, 1) = sorted_target(1, 1);
    j = 1;
    for i = 2:(NumberofTrainingData + NumberofTestingData)
        if sorted_target(1, i) ~= label(1, j)
            j = j + 1;
            label(1, j) = sorted_target(1, i);
        end
    end
    number_class = j;
    NumberofOutputNeurons = number_class;
       
    % Processing targets de entrenamiento
    temp_T = zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1, j) == T(1, i)
                break; 
            end
        end
        temp_T(j, i) = 1;
    end
    T = temp_T;

    % Processing targets de testeo
    temp_TV_T = zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1, j) == TV.T(1, i)
                break; 
            end
        end
        temp_TV_T(j, i) = 1;
    end
    TV.T = temp_TV_T;
end

%%%%%%%%%%% Entrenamiento
start_time_train = tic;  

% Generar pesos aleatorios
InputWeight = rand(NumberofHiddenNeurons, NumberofInputNeurons) * 2 - 1;
BiasofHiddenNeurons = rand(NumberofHiddenNeurons, 1) * 2 - 1;

% Calcular matriz H
tempH = InputWeight * P;
ind = ones(1, NumberofTrainingData);
BiasMatrix = BiasofHiddenNeurons(:, ind);
tempH = tempH + BiasMatrix;

% Función de activación
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        H = sin(tempH);    
    case {'hardlim'}
        H = double(hardlim(tempH));
    case {'tribas'}
        H = tribas(tempH);
    case {'radbas'}
        H = radbas(tempH);
end
clear tempH;

%%%%%%%%%%% Calcular pesos de salida con regularización
if (NumberofTrainingData > NumberofHiddenNeurons)
    OutputWeight = ((H * H' + speye(NumberofHiddenNeurons) / C) \ (H * T'));
else
    OutputWeight = H * ((H' * H + speye(size(T, 2)) / C) \ (T'));
end

TrainingTime_r = toc(start_time_train); 

%%%%%%%%%%% Salida de entrenamiento
Y = (H' * OutputWeight)';

if Elm_Type == REGRESSION
    trainMSE_r = mse(T - Y);
    TrainingAccuracy_r = sqrt(trainMSE_r);
    [trainR_r, ~, ~] = regression(T, Y);
    
    % R² estándar
    SS_res_train = sum((T - Y).^2);
    SS_tot_train = sum((T - mean(T)).^2);
    if SS_tot_train == 0
        trainR2_r = 1;
    else
        trainR2_r = 1 - SS_res_train / SS_tot_train;
    end
end
clear H;

%%%%%%%%%%% Testeo
start_time_test = tic;  %

tempH_test = InputWeight * TV.P;
ind = ones(1, NumberofTestingData);
BiasMatrix = BiasofHiddenNeurons(:, ind);
tempH_test = tempH_test + BiasMatrix;

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        H_test = sin(tempH_test);        
    case {'hardlim'}
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        H_test = tribas(tempH_test);        
    case {'radbas'}
        H_test = radbas(tempH_test);
end

TY_R = (H_test' * OutputWeight)';

TestingTime_r = toc(start_time_test);
if Elm_Type == REGRESSION
    testMSE_r = mse(TV.T - TY_R);
    TestingAccuracy_r = sqrt(testMSE_r);
    [testR_r, ~, ~] = regression(TV.T, TY_R);
    
    % R² estándar
    SS_res_test = sum((TV.T - TY_R).^2);
    SS_tot_test = sum((TV.T - mean(TV.T)).^2);
    if SS_tot_test == 0
        testR2_r = 1;
    else
        testR2_r = 1 - SS_res_test / SS_tot_test;
    end
end

%%%%%%%%%%% Clasificación
if Elm_Type == CLASSIFIER
    MissClassificationRate_Training = 0;
    MissClassificationRate_Testing = 0;

    for i = 1:size(T, 2)
        [~, label_index_expected] = max(T(:, i));
        [~, label_index_actual] = max(Y(:, i));
        if label_index_actual ~= label_index_expected
            MissClassificationRate_Training = MissClassificationRate_Training + 1;
        end
    end
    TrainingAccuracy_r = 1 - MissClassificationRate_Training / size(T, 2);
    
    for i = 1:size(TV.T, 2)
        [~, label_index_expected] = max(TV.T(:, i));
        [~, label_index_actual] = max(TY_R(:, i));
        if label_index_actual ~= label_index_expected
            MissClassificationRate_Testing = MissClassificationRate_Testing + 1;
        end
    end
    TestingAccuracy_r = 1 - MissClassificationRate_Testing / size(TV.T, 2);
end

% Guardar valores reales de testeo
TV_T = TV.T;

end
