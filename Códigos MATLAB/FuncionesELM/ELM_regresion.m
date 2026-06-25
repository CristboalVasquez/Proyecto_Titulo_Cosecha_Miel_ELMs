function [TrainingTime_e, TestingTime_e, TrainingAccuracy_e, TestingAccuracy_e,...
    rtrain_e, rtest_e, trainMSE_e, testMSE_e, trainR2_e, testR2_e, TY_test, TY_train, T, Y] = ...
    ELM_regresion(entradaentrenamiento, salidasentrenamiento, entredatesteo, salidastesteo, ...
    Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% ELM_regresion - Extreme Learning Machine para regresión
%
% Salidas:
%   TrainingTime_e    - Tiempo de entrenamiento (s)
%   TestingTime_e     - Tiempo de testeo (s)
%   TrainingAccuracy_e - RMSE de entrenamiento
%   TestingAccuracy_e  - RMSE de testeo
%   rtrain_e          - Correlación R de entrenamiento
%   rtest_e           - Correlación R de testeo
%   trainMSE_e        - MSE de entrenamiento
%   testMSE_e         - MSE de testeo
%   trainR2_e         - R² de entrenamiento
%   testR2_e          - R² de testeo
%   TY_test           - Predicciones de testeo
%   TY_train          - Predicciones de entrenamiento
%   T                 - Valores reales de entrenamiento
%   Y                 - Valores predichos de entrenamiento

% Definición de tipo de operación
REGRESSION = 0;
CLASSIFIER = 1;

% Preparar matrices de entrenamiento (transponer)
T = salidasentrenamiento';
P = entradaentrenamiento';

% Preparar matrices de testeo (transponer)
TV.T = salidastesteo';
TV.P = entredatesteo';

% Definir tamaños de datos y entradas
NumberofTrainingData = size(P, 2);
NumberofTestingData = size(TV.P, 2);
NumberofInputNeurons = size(P, 1);

% Iniciar contador de tiempo de entrenamiento
start_time_train = cputime;

% Inicializar pesos de entrada y sesgos de neuronas ocultas aleatoriamente
InputWeight = rand(NumberofHiddenNeurons, NumberofInputNeurons) * 2 - 1;
BiasofHiddenNeurons = rand(NumberofHiddenNeurons, 1) * 2 - 1;

% Calcular entradas a neuronas ocultas
tempH = InputWeight * P;
clear P;
ind = ones(1, NumberofTrainingData);
BiasMatrix = BiasofHiddenNeurons(:, ind);
tempH = tempH + BiasMatrix;

% Aplicar función de activación seleccionada
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

% Calcular pesos de salida utilizando pseudoinversa
OutputWeight = pinv(H') * T';

% Finalizar contador de tiempo de entrenamiento
end_time_train = cputime;
TrainingTime_e = end_time_train - start_time_train;

% Calcular salida del modelo en entrenamiento
Y = (H' * OutputWeight)';

% Calcular métricas para entrenamiento
if Elm_Type == REGRESSION
    trainMSE_e = mse(T - Y);
    TrainingAccuracy_e = sqrt(trainMSE_e);  % RMSE
    [rtrain_e, ~, ~] = regression(T, Y);    % R (correlación)
    
    % R² usando fórmula estándar: 1 - SS_res/SS_tot
    SS_res_train = sum((T - Y).^2);
    SS_tot_train = sum((T - mean(T)).^2);
    if SS_tot_train == 0
        trainR2_e = 1;  % Caso especial: todos los valores son iguales
    else
        trainR2_e = 1 - SS_res_train / SS_tot_train;
    end
end
clear H;

% Guardar predicciones de entrenamiento
TY_train = Y;

% Iniciar contador de tiempo de testeo
start_time_test = cputime;

% Calcular entradas de testeo para neuronas ocultas
tempH_test = InputWeight * TV.P;
clear TV.P;
ind = ones(1, NumberofTestingData);
BiasMatrix = BiasofHiddenNeurons(:, ind);
tempH_test = tempH_test + BiasMatrix;

% Aplicar función de activación a datos de testeo
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

% Calcular salida del modelo en testeo
TY_E = (H_test' * OutputWeight)';

% Finalizar contador de tiempo de testeo
end_time_test = cputime;
TestingTime_e = end_time_test - start_time_test;

% Calcular métricas para testeo
if Elm_Type == REGRESSION
    testMSE_e = mse(TV.T - TY_E);
    TestingAccuracy_e = sqrt(testMSE_e);     % RMSE
    [rtest_e, ~, ~] = regression(TV.T, TY_E); % R (correlación) - orden corregido
    
    % R² usando fórmula estándar: 1 - SS_res/SS_tot
    SS_res_test = sum((TV.T - TY_E).^2);
    SS_tot_test = sum((TV.T - mean(TV.T)).^2);
    if SS_tot_test == 0
        testR2_e = 1;
    else
        testR2_e = 1 - SS_res_test / SS_tot_test;
    end
end

% Guardar predicciones de testeo
TY_test = TY_E;

end
