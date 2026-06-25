clc; clear all; close all;

%% 1. LECTURA DE DATOS
fprintf('=== INICIANDO EXPERIMENTO DE REGRESIÓN (5-FOLD CV) ===\n');

base_datos = 'BaseDeDatos_201_muestras_con_cosecha.xlsx';
%
C_data = readcell(base_datos);
M_raw = str2double(strrep(string(C_data), ',', '.'));
if any(isnan(M_raw(1, :)))
    M = M_raw(2:end, :);
else
    M = M_raw;
end

% Filtro de seguridad borrar filas si falta un dato
M(any(isnan(M), 2), :) = []; 

datos_temp    = M(:, 1:34);    
datos_precip  = M(:, 35:45);   
datos_cosecha = M(:, 46);      

fprintf('ESTADÍSTICAS DEL DATASET ENRIQUECIDO\n');
fprintf('Cosecha - Min: %.2f, Max: %.2f, Media: %.2f, Desv. std: %.2f\n', ...
    min(datos_cosecha), max(datos_cosecha), mean(datos_cosecha), std(datos_cosecha));

numSamples = size(datos_cosecha, 1);
fprintf('Total de muestras válidas a entrenar: %d\n', numSamples);
%% 2. NORMALIZACIÓN [-1, 1]
fprintf('\nAplicando normalización global a variables climáticas...\n');

% Normalizar Temperaturas
mx_T = max(datos_temp, [], 1); 
mn_T = min(datos_temp, [], 1);
diff_T = mx_T - mn_T; 
diff_T(diff_T == 0) = 1; 
temp_norm = 2 .* (datos_temp - mn_T) ./ diff_T - 1;

% Normalizar Precipitaciones
mx_P = max(datos_precip, [], 1); 
mn_P = min(datos_precip, [], 1);
diff_P = mx_P - mn_P; 
diff_P(diff_P == 0) = 1;
precip_norm = 2 .* (datos_precip - mn_P) ./ diff_P - 1;

%% 3. CONSTRUCCIÓN DE LA TABLA Y EXPORTACIÓN AL EXCEL
fprintf('Construyendo la base de datos normalizada para App\n');

% Juntar las piezas: Entradas normalizadas + Salida (Cosecha) en crudo
M_norm = [temp_norm, precip_norm, datos_cosecha];

% Convertir la matriz de vuelta a formato 'table'
T_norm = array2table(M_norm);

% Asignar nombres limpios a las columnas (1 a 45 para clima, 46 para cosecha)
for i = 1:45
    T_norm.Properties.VariableNames{i} = ['Var_Clima_', num2str(i)];
end
T_norm.Properties.VariableNames{46} = 'Cosecha_kg';

fprintf('ESTADÍSTICAS DEL DATASET EXPORTADO (201 Muestras)\n');
fprintf('Cosecha (Original)  - Min: %.2f, Max: %.2f, Media: %.2f\n', ...
    min(datos_cosecha), max(datos_cosecha), mean(datos_cosecha));
fprintf('Temp (Normalizada)  - Min: %.2f, Max: %.2f\n', min(temp_norm(:)), max(temp_norm(:)));
fprintf('Precip (Normalizada)- Min: %.2f, Max: %.2f\n', min(precip_norm(:)), max(precip_norm(:)));

% Guardar archivo Excel para la App
timestamp = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
nombre_archivo_salida = sprintf('BaseDeDatos201_Normalizado_%s.xlsx', timestamp);
writetable(T_norm, nombre_archivo_salida);
fprintf('\n✓ EXPORTACIÓN EXITOSA: Archivo guardado como "%s"\n', nombre_archivo_salida);
%%  PREPROCESAMIENTO Y K-FOLD CROSS-VALIDATION
K = 5;
% rng(); % semilla para que los folds sean consistentes
c = cvpartition(numSamples, 'KFold', K);
fprintf('Validación Cruzada configurada a %d-Fold con %d muestras totales.\n', K, numSamples);

entrenamientoe_cv = cell(1, K);
testeoe_cv        = cell(1, K);
entrenamientos_cv = cell(1, K);
testeos_cv        = cell(1, K);

for j = 1:K
    idxTrain = training(c, j);
    idxTest  = test(c, j);
    
    % Normalización Anti-Leakage (Temperatura)
    t_tr = datos_temp(idxTrain, :); t_te = datos_temp(idxTest, :);
    mx_T = max(t_tr, [], 1); mn_T = min(t_tr, [], 1);
    diff_T = mx_T - mn_T; diff_T(diff_T == 0) = 1; 
    nt_tr = 2 .* (t_tr - mn_T) ./ diff_T - 1;
    nt_te = 2 .* (t_te - mn_T) ./ diff_T - 1;
    
    % Normalización Anti-Leakage (Precipitación)
    p_tr = datos_precip(idxTrain, :); p_te = datos_precip(idxTest, :);
    mx_P = max(p_tr, [], 1); mn_P = min(p_tr, [], 1);
    diff_P = mx_P - mn_P; diff_P(diff_P == 0) = 1;
    np_tr = 2 .* (p_tr - mn_P) ./ diff_P - 1;
    np_te = 2 .* (p_te - mn_P) ./ diff_P - 1;

    entrenamientoe_cv{j} = [nt_tr, np_tr];
    testeoe_cv{j}        = [nt_te, np_te];
    entrenamientos_cv{j} = datos_cosecha(idxTrain, :);
    testeos_cv{j}        = datos_cosecha(idxTest, :);
end

%% 2. HIPERPARÁMETROS DE BÚSQUEDA
C_vec  = 2.^(-20:2:20);
NO_vec_E = 10:10:160;
NO_vec_R = 10:10:160;
NO1_vec_MELM2 = 10:10:160;
NO2_vec_MELM2 = 10:10:160;
NO1_vec = 10:10:160; 
NO2_vec = 10:10:160; 
NO3_vec = 10:10:160;
rhoValue = 0.05; 
sigpara  = [1 1 1];
sigpara1 = [1 1 1];

%% 3. BÚSQUEDA HIPERPARÁMETROS ELM ESTÁNDAR
fprintf('\nBuscando hiperparámetros para ELM estándar...\n');
RMSE_test_vec_ELM  = zeros(1, length(NO_vec_E));
R_test_vec_ELM     = zeros(1, length(NO_vec_E));
R2_test_vec_ELM    = zeros(1, length(NO_vec_E));
RMSE_train_vec_ELM = zeros(1, length(NO_vec_E));
R_train_vec_ELM    = zeros(1, length(NO_vec_E));
R2_train_vec_ELM   = zeros(1, length(NO_vec_E));
Time_vec_ELM       = zeros(1, length(NO_vec_E));
Speed_vec_ELM      = zeros(1, length(NO_vec_E));

for jj = 1:length(NO_vec_E)
    NO = NO_vec_E(jj);
    rmse_train_vals  = zeros(1, K);
    training_times   = zeros(1, K);
    pred_test_all    = zeros(1, numSamples);
    real_test_all    = zeros(1, numSamples);
    pred_train_all   = [];
    real_train_all   = [];
    
    idx_test_start = 1;
    for j = 1:K
        [Time_e, ~, Acc_e, ~, ~, ~, ~, ~, ~, ~, TY_test, TY_train, ~, ~] = ...
            ELM_regresion(entrenamientoe_cv{j}, entrenamientos_cv{j}, testeoe_cv{j}, testeos_cv{j}, 0, NO, 'sin');
                          
        rmse_train_vals(j) = Acc_e;
        training_times(j)  = Time_e;
        
        n_test = length(testeos_cv{j});
        idx_test_end = idx_test_start + n_test - 1;
        pred_test_all(idx_test_start:idx_test_end) = TY_test(:)';
        real_test_all(idx_test_start:idx_test_end) = testeos_cv{j}(:)';
        idx_test_start = idx_test_end + 1;
        
        pred_train_all = [pred_train_all, TY_train(:)'];
        real_train_all = [real_train_all, entrenamientos_cv{j}(:)'];
    end
    
    err_test = real_test_all - pred_test_all;
    RMSE_test_vec_ELM(jj) = sqrt(mean(err_test.^2));
    SS_res_test = sum(err_test.^2);
    SS_tot_test = sum((real_test_all - mean(real_test_all)).^2);
    R2_test_vec_ELM(jj) = 1 - SS_res_test/SS_tot_test;
    if std(real_test_all) > 0 && std(pred_test_all) > 0
        R_corr_test = corrcoef(real_test_all, pred_test_all);
        R_test_vec_ELM(jj) = R_corr_test(1,2);
    else
        R_test_vec_ELM(jj) = NaN;
    end
    
    RMSE_train_vec_ELM(jj) = mean(rmse_train_vals);
    err_train = real_train_all - pred_train_all;
    SS_res_train = sum(err_train.^2);
    SS_tot_train = sum((real_train_all - mean(real_train_all)).^2);
    R2_train_vec_ELM(jj) = 1 - SS_res_train/SS_tot_train;
    if std(real_train_all) > 0 && std(pred_train_all) > 0
        R_corr_train = corrcoef(real_train_all, pred_train_all);
        R_train_vec_ELM(jj) = R_corr_train(1,2);
    else
        R_train_vec_ELM(jj) = NaN;
    end
    
    Time_vec_ELM(jj) = mean(training_times);
    Speed_vec_ELM(jj) = numSamples / mean(training_times);
end
fprintf(' ELM estándar completada \n');

%% 4. BÚSQUEDA HIPERPARÁMETROS ELM2REGU
fprintf('\nBuscando hiperparámetros para ELM2REGU...\n');
RMSE_test_matrix_REGU = zeros(length(C_vec), length(NO_vec_R));
R_test_matrix_REGU    = zeros(length(C_vec), length(NO_vec_R));
R2_test_matrix_REGU   = zeros(length(C_vec), length(NO_vec_R));
Time_matrix_REGU      = zeros(length(C_vec), length(NO_vec_R));
Speed_matrix_REGU     = zeros(length(C_vec), length(NO_vec_R));

for jjj = 1:length(C_vec)
    C_current = C_vec(jjj);
    for jj = 1:length(NO_vec_R)
        NO = NO_vec_R(jj);
        training_times   = zeros(1, K);
        pred_test_all    = zeros(1, numSamples); 
        real_test_all    = zeros(1, numSamples); 
        
        idx_test_start = 1;
        for j = 1:K
            [Time_r, ~, ~, ~, ~, ~, ~, ~, ~, ~, TY_R, ~, ~, ~] = ...
                ELM2REGU(entrenamientos_cv{j}', entrenamientoe_cv{j}', testeos_cv{j}', testeoe_cv{j}', 0, NO, 'sin', C_current);
            
            training_times(j)  = Time_r;
            n_test = length(testeos_cv{j});
            idx_test_end = idx_test_start + n_test - 1;
            pred_test_all(idx_test_start:idx_test_end) = TY_R(:)';
            real_test_all(idx_test_start:idx_test_end) = testeos_cv{j}(:)';
            idx_test_start = idx_test_end + 1;
        end
        
        err_test = real_test_all - pred_test_all;
        RMSE_test_matrix_REGU(jjj, jj) = sqrt(mean(err_test.^2));
        SS_res_test = sum(err_test.^2);
        SS_tot_test = sum((real_test_all - mean(real_test_all)).^2);
        R2_test_matrix_REGU(jjj, jj) = 1 - SS_res_test/SS_tot_test;
        
        if std(real_test_all) > 0 && std(pred_test_all) > 0
            R_corr_test = corrcoef(real_test_all, pred_test_all);
            R_test_matrix_REGU(jjj, jj) = R_corr_test(1,2);
        else
            R_test_matrix_REGU(jjj, jj) = NaN;
        end
        Time_matrix_REGU(jjj, jj) = mean(training_times);
        Speed_matrix_REGU(jjj, jj) = numSamples / mean(training_times);
    end
    fprintf('ELM2REGU: C = 2^%d completado\n', log2(C_current));
end
fprintf(' ELM2REGU completada\n');

%% 5. BÚSQUEDA MELM2 (2 capas)
fprintf('\nBuscando hiperparámetros para MELM2 (2 capas)...\n');
RMSE_test_matrix_MELM2  = zeros(length(NO1_vec_MELM2), length(NO2_vec_MELM2));
R_test_matrix_MELM2     = zeros(length(NO1_vec_MELM2), length(NO2_vec_MELM2));
R2_test_matrix_MELM2    = zeros(length(NO1_vec_MELM2), length(NO2_vec_MELM2));
Time_matrix_MELM2       = zeros(length(NO1_vec_MELM2), length(NO2_vec_MELM2));

C_fijo_MELM2 = 1;
for idx1 = 1:length(NO1_vec_MELM2)
    NO1 = NO1_vec_MELM2(idx1);
    fprintf('MELM2: NO1 = %d (%d/%d)\n', NO1, idx1, length(NO1_vec_MELM2));
    for idx2 = 1:length(NO2_vec_MELM2)
        NO2 = NO2_vec_MELM2(idx2);
        training_times  = zeros(1, K);
        pred_test_all   = zeros(1, numSamples);
        real_test_all   = zeros(1, numSamples);
        
        idx_test_start = 1;
        for j = 1:K
            [Time_m, ~, ~, ~, ~, ~, ~, ~, ~, ~, TY_M, ~, ~, ~] = ...
                MELM_MNIST25(entrenamientos_cv{j}, entrenamientoe_cv{j}, testeos_cv{j}, testeoe_cv{j}, 0, 0, 0, 2, ...
                             [NO1 NO2], [C_fijo_MELM2 C_fijo_MELM2 C_fijo_MELM2], rhoValue, sigpara(1:2), sigpara1(1:2));
            training_times(j) = Time_m;
            n_test = length(testeos_cv{j});
            idx_test_end = idx_test_start + n_test - 1;
            pred_test_all(idx_test_start:idx_test_end)  = TY_M(:)';
            real_test_all(idx_test_start:idx_test_end)  = testeos_cv{j}(:)';
            idx_test_start = idx_test_end + 1;
        end
        
        err_test = real_test_all - pred_test_all;
        RMSE_test_matrix_MELM2(idx1, idx2) = sqrt(mean(err_test.^2)); 
        if std(real_test_all) > 0 && std(pred_test_all) > 0
            mat_corr = corrcoef(real_test_all, pred_test_all);
            R_test_matrix_MELM2(idx1, idx2) = mat_corr(1,2);
        else
            R_test_matrix_MELM2(idx1, idx2) = NaN;
        end
        SS_res = sum(err_test.^2);
        SS_tot = sum((real_test_all - mean(real_test_all)).^2);
        R2_test_matrix_MELM2(idx1, idx2) = 1 - SS_res/SS_tot;
        Time_matrix_MELM2(idx1, idx2) = mean(training_times);
    end
end
fprintf('Búsqueda MELM2 completada \n');

%% 6. BÚSQUEDA MELM3 (3 capas)
fprintf('\nBuscando hiperparámetros para MELM (3 capas)\n');
RMSE_test_matrix_MELM3 = zeros(length(NO1_vec), length(NO2_vec), length(NO3_vec));
R_test_matrix_MELM3    = zeros(length(NO1_vec), length(NO2_vec), length(NO3_vec));
R2_test_matrix_MELM3   = zeros(length(NO1_vec), length(NO2_vec), length(NO3_vec));
RMSE_train_matrix_MELM3= zeros(length(NO1_vec), length(NO2_vec), length(NO3_vec));
R_train_matrix_MELM3   = zeros(length(NO1_vec), length(NO2_vec), length(NO3_vec));
R2_train_matrix_MELM3  = zeros(length(NO1_vec), length(NO2_vec), length(NO3_vec));
Time_matrix_MELM3      = zeros(length(NO1_vec), length(NO2_vec), length(NO3_vec));

C_fijo_MELM = 1;
contador_total = 0;
total_combinaciones = length(NO1_vec) * length(NO2_vec) * length(NO3_vec);

for idx1 = 1:length(NO1_vec)
    NO1 = NO1_vec(idx1);
    fprintf('\n--- MELM3: NO1 = %d (%d/%d) ---\n', NO1, idx1, length(NO1_vec));
    for idx2 = 1:length(NO2_vec)
        NO2 = NO2_vec(idx2);
        for idx3 = 1:length(NO3_vec)
            NO3 = NO3_vec(idx3);
            contador_total = contador_total + 1;
            
            if mod(contador_total, 50) == 0 || contador_total == 1
                porcentaje = 100 * contador_total / total_combinaciones;
                fprintf(' NO2=%2d, NO3=%2d | Progreso: %d/%d (%.1f%%)\n', ...
                NO2, NO3, contador_total, total_combinaciones, porcentaje);
            end
            
            rmse_train_vals = zeros(1, K);
            training_times = zeros(1, K);
            pred_test_all = zeros(1, numSamples); real_test_all = zeros(1, numSamples);
            pred_train_all = []; real_train_all = [];
            
            idx_test_start = 1;
            for j = 1:K
                [Time_m, ~, ~, ~, ~, ~, ~, ~, ~, ~, TY_M, Y_M, ~, ~] = ...
                    MELM_MNIST25(entrenamientos_cv{j}, entrenamientoe_cv{j}, testeos_cv{j}, testeoe_cv{j}, 0, 0, 0, 3, ...
                                 [NO1 NO2 NO3], [C_fijo_MELM C_fijo_MELM C_fijo_MELM C_fijo_MELM], rhoValue, sigpara, sigpara1);
                                 
                rmse_train_vals(j) = sqrt(mean((entrenamientos_cv{j}' - Y_M).^2));
                training_times(j) = Time_m;
                
                n_test = length(testeos_cv{j});
                pred_test_all(idx_test_start:idx_test_start+n_test-1) = TY_M(:)';
                real_test_all(idx_test_start:idx_test_start+n_test-1) = testeos_cv{j}(:)';
                idx_test_start = idx_test_start + n_test;
                
                pred_train_all = [pred_train_all, Y_M(:)'];
                real_train_all = [real_train_all, entrenamientos_cv{j}(:)'];
            end
            
            RMSE_train_matrix_MELM3(idx1, idx2, idx3) = mean(rmse_train_vals);
            err_test = real_test_all - pred_test_all;
            RMSE_test_matrix_MELM3(idx1, idx2, idx3) = sqrt(mean(err_test.^2));
            
            if std(real_test_all) > 0 && std(pred_test_all) > 0
                mat_corr = corrcoef(real_test_all, pred_test_all);
                R_test_matrix_MELM3(idx1, idx2, idx3) = mat_corr(1,2);
            else
                R_test_matrix_MELM3(idx1, idx2, idx3) = NaN;
            end
            
            SS_res = sum(err_test.^2); SS_tot = sum((real_test_all - mean(real_test_all)).^2);
            R2_test_matrix_MELM3(idx1, idx2, idx3) = 1 - SS_res/SS_tot;
            
            if std(real_train_all) > 0 && std(pred_train_all) > 0
                mat_corr_tr = corrcoef(real_train_all, pred_train_all);
                R_train_matrix_MELM3(idx1, idx2, idx3) = mat_corr_tr(1,2);
            else
                R_train_matrix_MELM3(idx1, idx2, idx3) = NaN;
            end
            
            err_train = real_train_all - pred_train_all;
            SS_res_tr = sum(err_train.^2); SS_tot_tr = sum((real_train_all - mean(real_train_all)).^2);
            R2_train_matrix_MELM3(idx1, idx2, idx3) = 1 - SS_res_tr/SS_tot_tr;
            Time_matrix_MELM3(idx1, idx2, idx3) = mean(training_times);
        end
    end
end
fprintf('\n Búsqueda MELM3 completada \n');

%% 7. IMPRESIÓN DE HIPERPARÁMETROS ÓPTIMOS
% Extracciones de índices basados en RMSE
[min_rmse_ELM, idx_best_ELM] = min(RMSE_test_vec_ELM);
mejorNO_ELM = NO_vec_E(idx_best_ELM);

[min_rmse_REGU, idx_regu] = min(RMSE_test_matrix_REGU(:));
[idx_c, idx_no] = ind2sub(size(RMSE_test_matrix_REGU), idx_regu);
mejorNO_REGU = NO_vec_R(idx_no);
mejorC_REGU  = C_vec(idx_c);

[min_rmse_MELM2, idx_melm2] = min(RMSE_test_matrix_MELM2(:));
[idx1_melm2, idx2_melm2] = ind2sub(size(RMSE_test_matrix_MELM2), idx_melm2);
mejorNO1_MELM2 = NO1_vec_MELM2(idx1_melm2); mejorNO2_MELM2 = NO2_vec_MELM2(idx2_melm2);

[min_rmse_MELM3, idxMELM3] = min(RMSE_test_matrix_MELM3(:));
[idx1_star, idx2_star, idx3_star] = ind2sub(size(RMSE_test_matrix_MELM3), idxMELM3);
mejorNO1_MELM3 = NO1_vec(idx1_star); mejorNO2_MELM3 = NO2_vec(idx2_star); mejorNO3_MELM3 = NO3_vec(idx3_star);

fprintf('\n HIPERPARÁMETROS ÓPTIMOS SELECCIONADOS (SEGÚN RMSE)\n');

% 1. ELM Estándar
fprintf('\n--- ELM estándar: NO=%d ---\n', mejorNO_ELM);
fprintf('  Test:  RMSE=%.4f, R=%.4f, R^2=%.4f\n', ...
    RMSE_test_vec_ELM(idx_best_ELM), R_test_vec_ELM(idx_best_ELM), R2_test_vec_ELM(idx_best_ELM));
fprintf('  Train: RMSE=%.4f, R=%.4f, R^2=%.4f\n', ...
    RMSE_train_vec_ELM(idx_best_ELM), R_train_vec_ELM(idx_best_ELM), R2_train_vec_ELM(idx_best_ELM));
fprintf('  Tiempo=%.6f s, Velocidad=%.2f muestras/s\n', ...
    Time_vec_ELM(idx_best_ELM), Speed_vec_ELM(idx_best_ELM));

% 2. ELM2REGU
fprintf('\n--- ELM2REGU: C=2^%d (%.4e), NO=%d ---\n', log2(mejorC_REGU), mejorC_REGU, mejorNO_REGU);
fprintf('  Test:  RMSE=%.4f, R=%.4f, R^2=%.4f\n', ...
    RMSE_test_matrix_REGU(idx_c, idx_no), R_test_matrix_REGU(idx_c, idx_no), R2_test_matrix_REGU(idx_c, idx_no));
fprintf('  Train: RMSE=%.4f, R=%.4f, R^2=%.4f\n', ...
    RMSE_train_REGU_rec, R_train_REGU_rec, R2_train_REGU_rec);
fprintf('  Tiempo=%.6f s, Velocidad=%.2f muestras/s\n', ...
    Time_matrix_REGU(idx_c, idx_no), Speed_matrix_REGU(idx_c, idx_no));

% 3. MELM2
fprintf('\n--- MELM2: NO1=%d, NO2=%d ---\n', mejorNO1_MELM2, mejorNO2_MELM2);
fprintf('  Test:  RMSE=%.4f, R=%.4f, R^2=%.4f\n', ...
    RMSE_test_matrix_MELM2(idx1_melm2, idx2_melm2), R_test_matrix_MELM2(idx1_melm2, idx2_melm2), R2_test_matrix_MELM2(idx1_melm2, idx2_melm2));
fprintf('  Train: RMSE=%.4f, R=%.4f, R^2=%.4f\n', ...
    RMSE_train_MELM2_rec, R_train_MELM2_rec, R2_train_MELM2_rec);
fprintf('  Tiempo=%.6f s, Velocidad=%.2f muestras/s\n', ...
    Time_matrix_MELM2(idx1_melm2, idx2_melm2), numSamples / Time_matrix_MELM2(idx1_melm2, idx2_melm2));

% 4. MELM3
fprintf('\n--- MELM3: NO1=%d, NO2=%d, NO3=%d ---\n', mejorNO1_MELM3, mejorNO2_MELM3, mejorNO3_MELM3);
fprintf('  Test:  RMSE=%.4f, R=%.4f, R^2=%.4f\n', ...
    RMSE_test_matrix_MELM3(idx1_star, idx2_star, idx3_star), R_test_matrix_MELM3(idx1_star, idx2_star, idx3_star), R2_test_matrix_MELM3(idx1_star, idx2_star, idx3_star));
fprintf('  Train: RMSE=%.4f, R=%.4f, R^2=%.4f\n', ...
    RMSE_train_matrix_MELM3(idx1_star, idx2_star, idx3_star), R_train_matrix_MELM3(idx1_star, idx2_star, idx3_star), R2_train_matrix_MELM3(idx1_star, idx2_star, idx3_star));
fprintf('  Tiempo=%.6f s, Velocidad=%.2f muestras/s\n', ...
    Time_matrix_MELM3(idx1_star, idx2_star, idx3_star), numSamples / Time_matrix_MELM3(idx1_star, idx2_star, idx3_star));


%% 8. VISUALIZACIONES
fprintf('\n Generando visualizaciones \n');

% 8.1 ELM2REGU
[min_rmse_test, max_rmse_test, idx_c_min_rmse, idx_no_min_rmse, idx_c_max_rmse, idx_no_max_rmse] = calcular_extremos(RMSE_test_matrix_REGU);
[min_r_test, max_r_test, idx_c_min_r, idx_no_min_r, idx_c_max_r, idx_no_max_r] = calcular_extremos(R_test_matrix_REGU);
[NO_grid, C_grid] = meshgrid(NO_vec_R, log2(C_vec));
resultado_text = sprintf(['\\bf Óptimo RMSE:\\rm NO=%d, C=2^{%d}\n\n' ...
    '\\bf Test:\\rm\n  RMSE: %.2f [%.2f,%.2f]\n  R: %.3f [%.3f,%.3f]'], ...
    mejorNO_REGU, log2(mejorC_REGU), ...
    RMSE_test_matrix_REGU(idx_c_min_rmse, idx_no_min_rmse), ...
    min_rmse_test, max_rmse_test, ...
    R_test_matrix_REGU(idx_c_min_rmse, idx_no_min_rmse), min_r_test, max_r_test);

figure('Position', [100 100 1200 400], 'Name', 'ELM2REGU');
subplot(1,3,1);
plot_contour_metric(NO_grid, C_grid, RMSE_test_matrix_REGU, 'A', 'RMSE (Kg/Año)', ...
    NO_vec_R, C_vec, idx_no_min_rmse, idx_c_min_rmse, idx_no_max_rmse, idx_c_max_rmse, 'gx', 'rx', false);
subplot(1,3,2);
plot_contour_metric(NO_grid, C_grid, R_test_matrix_REGU, 'B', 'Coeficiente de Correlación (R)', ...
    NO_vec_R, C_vec, idx_no_max_r, idx_c_max_r, idx_no_min_r, idx_c_min_r, 'gx', 'rx', true);
subplot(1,3,3);
axis off;
text(0.05, 0.5, resultado_text, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', ...
    'FontSize', 10, 'FontName', 'Courier New', 'BackgroundColor', [1 1 0.9], 'EdgeColor', 'k', 'Margin', 14, 'LineWidth', 1.5);

% 8.2 ELM ESTÁNDAR
texto_optimo_ELM = sprintf(['Óptimo: NO=%d\nTest:  RMSE=%.2f, R=%.3f\n' ...
    'Train: RMSE=%.2f, R=%.3f'], ...
    mejorNO_ELM, RMSE_test_vec_ELM(idx_best_ELM), R_test_vec_ELM(idx_best_ELM), ...
    RMSE_train_vec_ELM(idx_best_ELM), R_train_vec_ELM(idx_best_ELM));

figure('Position', [150 150 1000 450], 'Name', 'ELM Estándar');
metrics = {RMSE_test_vec_ELM, RMSE_train_vec_ELM, 'RMSE (Kg/Año)', 'A'; ...
R_test_vec_ELM, R_train_vec_ELM, 'Coeficiente de Correlación (R)', 'B'};
for i = 1:2
    subplot(1,3,i);
    plot(NO_vec_E, metrics{i,1}, '-ob', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'Test'); hold on;
    plot(NO_vec_E, metrics{i,2}, '-sg', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'Train');
    plot(mejorNO_ELM, metrics{i,1}(idx_best_ELM), 'rx', 'MarkerSize', 15, 'LineWidth', 3, 'DisplayName', 'Ópt Test');
    plot(mejorNO_ELM, metrics{i,2}(idx_best_ELM), 'r+', 'MarkerSize', 15, 'LineWidth', 3, 'DisplayName', 'Ópt Train');
    xlabel('Neuronas Ocultas (NO)'); ylabel(metrics{i,3}); title(metrics{i,4}); legend('Location', 'best'); grid on;
end
subplot(1,3,3); axis off;
text(0.05, 0.5, texto_optimo_ELM, 'FontSize', 13, 'FontName', 'Courier', 'VerticalAlignment', 'middle', ...
    'HorizontalAlignment', 'left', 'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'k', 'Margin', 14, 'LineWidth', 1.5);
title('Resultados Óptimos', 'FontSize', 12);

% 8.3 MELM2
[NO2_grid_MELM2, NO1_grid_MELM2] = meshgrid(NO2_vec_MELM2, NO1_vec_MELM2);
[~, max_r_melm2, ~, ~, idx1_max_r_melm2, idx2_max_r_melm2] = calcular_extremos(R_test_matrix_MELM2);

resultado_text_melm2 = sprintf(['\\bf Óptimos MELM2:\\rm\n\n' ...
    '\\bf RMSE Test:\\rm\n  Mejor: %.2f\n  NO1=%d, NO2=%d\n\n' ...
    '\\bf R Test:\\rm\n  Mejor: %.3f\n  NO1=%d, NO2=%d'], ...
    min_rmse_MELM2, NO1_vec_MELM2(idx1_melm2), NO2_vec_MELM2(idx2_melm2), ...
    max_r_melm2, NO1_vec_MELM2(idx1_max_r_melm2), NO2_vec_MELM2(idx2_max_r_melm2));

figure('Position', [200 200 1200 400], 'Name', 'MELM2');
subplot(1,3,1);
contourf(NO2_grid_MELM2, NO1_grid_MELM2, RMSE_test_matrix_MELM2, 15, 'LineColor', 'none');
hold on;
plot(NO2_vec_MELM2(idx2_melm2), NO1_vec_MELM2(idx1_melm2), 'xg', 'MarkerSize', 14, 'LineWidth', 3); %,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
hcb1 = colorbar;
ylabel(hcb1, 'RMSE (kg/año)', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Neuronas Ocultas Capa 2 (NO2)', 'FontSize', 11);
ylabel('Neuronas Ocultas Capa 1 (NO1)', 'FontSize', 11);
title('A', 'FontSize', 11);
grid on;
axis tight;

subplot(1,3,2);
contourf(NO2_grid_MELM2, NO1_grid_MELM2, R_test_matrix_MELM2, 15, 'LineColor', 'none');
hold on;
plot(NO2_vec_MELM2(idx2_max_r_melm2), NO1_vec_MELM2(idx1_max_r_melm2), 'xg', 'MarkerSize', 14, 'LineWidth', 3); %,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
hcb2 = colorbar;
ylabel(hcb2, 'Coeficiente de Correlación (R)', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Neuronas Ocultas Capa 2 (NO2)', 'FontSize', 11);
ylabel('Neuronas Ocultas Capa 1 (NO1)', 'FontSize', 11);
title('B', 'FontSize', 11);
grid on;
axis tight;

subplot(1,3,3);
axis off;
text(0.05, 0.5, resultado_text_melm2, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', ...
    'FontSize', 10, 'FontName', 'Courier New', 'BackgroundColor', [0.95 1 0.95], 'EdgeColor', 'k', 'Margin', 14, 'LineWidth', 1.5);
title('Hiperparámetros Óptimos', 'FontSize', 12, 'FontWeight', 'bold');

% 8.4 MELM3 (Gráficos 3d)
[idx1_all, idx2_all, idx3_all] = ndgrid(1:length(NO1_vec), 1:length(NO2_vec), 1:length(NO3_vec));
NO1_all = NO1_vec(idx1_all(:)); NO2_all = NO2_vec(idx2_all(:)); NO3_all = NO3_vec(idx3_all(:));
RMSE_all = RMSE_test_matrix_MELM3(:); R_all = R_test_matrix_MELM3(:);

valid_idx = ~isnan(RMSE_all);
NO1_valid  = NO1_all(valid_idx); NO2_valid  = NO2_all(valid_idx); NO3_valid  = NO3_all(valid_idx);
RMSE_valid = RMSE_all(valid_idx); R_valid = R_all(valid_idx);

figure('Position', [250 250 1200 500], 'Name', 'MELM3 3D');
subplot(1,2,1);
scatter3(NO1_valid, NO2_valid, NO3_valid, 80, RMSE_valid, 'filled', 'MarkerEdgeColor', 'k');
hold on;
rmse_min = min(RMSE_valid);
rmse_max = prctile(RMSE_valid, 95);
caxis([rmse_min, rmse_max]);
cmap = colormap;
num_colors = size(cmap, 1);
rmse_norm = max(0, min(1, (min_rmse_MELM3 - rmse_min) / (rmse_max - rmse_min)));
color_optimo_rmse = cmap(round(rmse_norm * (num_colors - 1)) + 1, :);
plot3(mejorNO1_MELM3, mejorNO2_MELM3, mejorNO3_MELM3, 'p', 'MarkerSize', 24, 'LineWidth', 3, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'r');
hcb1 = colorbar; ylabel(hcb1, 'RMSE (kg/año)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('NO1', 'FontSize', 11);
ylabel('NO2', 'FontSize', 11);
zlabel('NO3', 'FontSize', 11);
title(sprintf('A'), 'FontSize', 11);
grid on;
view(45, 30);

subplot(1,2,2);
scatter3(NO1_valid, NO2_valid, NO3_valid, 80, R_valid, 'filled', 'MarkerEdgeColor', 'k'); hold on;
r_min = prctile(R_valid, 5); r_max = max(R_valid); caxis([r_min, r_max]);
cmap = colormap; num_colors = size(cmap, 1);
r_norm = max(0, min(1, (R_test_matrix_MELM3(idx1_star, idx2_star, idx3_star) - r_min) / (r_max - r_min)));
color_optimo_r = cmap(round(r_norm * (num_colors - 1)) + 1, :);
plot3(mejorNO1_MELM3, mejorNO2_MELM3, mejorNO3_MELM3, 'p', 'MarkerSize', 24, 'LineWidth', 3, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'r');
hcb2 = colorbar; ylabel(hcb2, 'Coeficiente de Correlación (R)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('NO1', 'FontSize', 11); ylabel('NO2', 'FontSize', 11); zlabel('NO3', 'FontSize', 11);
title(sprintf('B'), 'FontSize', 11); grid on; view(45, 30);

fprintf('Visualizaciones Completadas \n');

%% 9. TABLA COMPARATIVA FINAL + EXPORTAR A EXCEL
fprintf('\n=== TABLA COMPARATIVA FINAL ===\n');
modelos = {'ELM estándar'; 'ELM2REGU'; 'MELM2 (2 capas)'; 'MELM3 (3 capas)'};

hiperparam = {
    sprintf('NO=%d', mejorNO_ELM);
    sprintf('NO=%d, C=2^{%d}', mejorNO_REGU, log2(mejorC_REGU));
    sprintf('NO1=%d, NO2=%d', mejorNO1_MELM2, mejorNO2_MELM2);
    sprintf('NO1=%d, NO2=%d, NO3=%d', mejorNO1_MELM3, mejorNO2_MELM3, mejorNO3_MELM3)
};

rmse_test = [min_rmse_ELM; RMSE_test_matrix_REGU(idx_c, idx_no); min_rmse_MELM2; min_rmse_MELM3];
r_test    = [R_test_vec_ELM(idx_best_ELM); R_test_matrix_REGU(idx_c, idx_no); R_test_matrix_MELM2(idx1_melm2, idx2_melm2); R_test_matrix_MELM3(idx1_star, idx2_star, idx3_star)];

tiempo_train = [Time_vec_ELM(idx_best_ELM); Time_matrix_REGU(idx_c, idx_no); Time_matrix_MELM2(idx1_melm2, idx2_melm2); Time_matrix_MELM3(idx1_star, idx2_star, idx3_star)];
velocidad_test = numSamples ./ tiempo_train;

tabla_comparativa = table(modelos, hiperparam, rmse_test, r_test, tiempo_train, velocidad_test, ...
    'VariableNames', {'Modelo', 'Hiperparametros_Optimos', 'RMSE_Test', 'R_Test_Asociado', 'Tiempo_Train_s', 'Velocidad_mps'});

for i = 3:width(tabla_comparativa)
    if isnumeric(tabla_comparativa{:,i})
        tabla_comparativa{:,i} = round(tabla_comparativa{:,i}, 4);
    end
end

fprintf('\n%-20s | %-45s | %-10s | %-15s\n', 'Modelo', 'Hiperparámetros (Min RMSE)', 'RMSE Test', 'R Asociado');
fprintf('%s\n', repmat('-', 1, 95));
for r = 1:height(tabla_comparativa)
    fprintf('%-20s | %-45s | %-10.4f | %-15.4f\n', ...
        tabla_comparativa.Modelo{r}, tabla_comparativa.Hiperparametros_Optimos{r}, ...
        tabla_comparativa.RMSE_Test(r), tabla_comparativa.R_Test_Asociado(r));
end

timestamp = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
nombre_archivo = sprintf('Comparativa_Final_N201_%s.xlsx', timestamp);
writetable(tabla_comparativa, nombre_archivo, 'Sheet', 'Resumen_Optimos_RMSE');
fprintf('\n Archivo Excel guardado: %s\n', nombre_archivo);

%% FUNCIONES AUXILIARES
function [val_min, val_max, idx_c_min, idx_no_min, idx_c_max, idx_no_max] = calcular_extremos(matriz)
    [val_min, idx_min] = min(matriz(:)); [idx_c_min, idx_no_min] = ind2sub(size(matriz), idx_min);
    [val_max, idx_max] = max(matriz(:)); [idx_c_max, idx_no_max] = ind2sub(size(matriz), idx_max);
end

function plot_contour_metric(NO_grid, C_grid, matriz, titulo, ylabel_txt, NO_vec, C_vec, ...
                             idx_no_mejor, idx_c_mejor, idx_no_peor, idx_c_peor, marker_mejor, marker_peor, invertir)
    val_min = min(matriz(:)); val_max = max(matriz(:)); niveles = linspace(val_min, val_max, 15);
    contourf(NO_grid, C_grid, matriz, niveles, 'LineColor', 'none'); hold on;
    hcb = colorbar; ylabel(hcb, ylabel_txt, 'FontSize', 10);
    h_mejor = plot(NO_vec(idx_no_mejor), log2(C_vec(idx_c_mejor)), marker_mejor, 'MarkerSize', 15, 'LineWidth', 3);
    h_peor  = plot(NO_vec(idx_no_peor),  log2(C_vec(idx_c_peor)),  marker_peor,  'MarkerSize', 15, 'LineWidth', 3);
    if invertir
        legend([h_mejor, h_peor], 'Máximo (mejor)', 'Mínimo (peor)', 'Location', 'best');
    else
        legend([h_mejor, h_peor], 'Mínimo (mejor)', 'Máximo (peor)', 'Location', 'best'); 
    end
    xlabel('Neuronas Ocultas (NO)'); ylabel('log_2(C)'); title(titulo); grid on;
end