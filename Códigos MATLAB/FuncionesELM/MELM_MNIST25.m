function [TrainingTime_m, TestingTime_m, TrainingAccuracy_m, TestingAccuracy_m, ...
    trainR_m, testR_m, trainMSE_m, testMSE_m, trainR2_m, testR2_m, TY_M, Y_M, TV_T, T] = ...
    MELM_MNIST25(salidas_entrenamiento, entradas_entrenamiento, salidas_testeo, entradas_testeo, ...
    Elm_Type, Testcode, TrainDataSize, TotalLayers, HiddernNeurons, C1, rhoValue, sigpara, sigpara1)

    REGRESSION = 0;
    CLASSIFIER = 1;

    % Inicializar salidas
    trainR_m = NaN;
    testR_m  = NaN;
    trainMSE_m = NaN;
    testMSE_m  = NaN;
    trainR2_m  = NaN;
    testR2_m   = NaN;

    % GUARDAR COPIAS REALES EN KG
    T_real_train = salidas_entrenamiento';
    T_real_test  = salidas_testeo';

    % Preparar datos de entrenamiento
    P = entradas_entrenamiento';
    T = salidas_entrenamiento';

    if TrainDataSize ~= 0
        rand_sequence = randperm(TrainDataSize);
        P = P(:, rand_sequence);
        T = T(:, rand_sequence);
        T_real_train = T_real_train(:, rand_sequence);
    end 

    TV.T = salidas_testeo';
    NumberofTrainingData = size(P, 2);
    NumberofTestingData  = size(TV.T, 2);
    NumberofInputNeurons = size(P, 1);

    % Clasificación (no usado en regresión)
    if Elm_Type == CLASSIFIER
        sorted_target = sort(cat(2, T, TV.T), 2);
        label = sorted_target(1,1);
        j = 1;
        for i = 2:(NumberofTrainingData + NumberofTestingData)
            if sorted_target(1,i) ~= label(j)
                j = j + 1;
                label(j) = sorted_target(1,i);
            end
        end
        number_class = j;
        NumberofOutputNeurons = number_class;

        temp_T = zeros(NumberofOutputNeurons, NumberofTrainingData);
        for i = 1:NumberofTrainingData
            for jj = 1:number_class
                if label(jj) == T(1,i), break; end
            end
            temp_T(jj,i) = 1;
        end
        T = temp_T*2 - 1;

        temp_TV_T = zeros(NumberofOutputNeurons, NumberofTestingData);
        for i = 1:NumberofTestingData
            for jj = 1:number_class
                if label(jj) == TV.T(1,i), break; end
            end
            temp_TV_T(jj,i) = 1;
        end
        TV.T = temp_TV_T*2 - 1;
    end

    % ========== CONSTRUCCIÓN DE LA ARQUITECTURA ==========
    train_time = tic;
    no_Layers = TotalLayers;
    stack = cell(no_Layers + 1, 1);
    
    % Construir vector HN (arquitectura de neuronas)
    HN = [NumberofInputNeurons, HiddernNeurons];  % Ej: [45, NO1, NO2, NO3]
    
    % Expandir vectores sigscale si son muy cortos
    if length(sigpara) < no_Layers
        sigscale = repmat(sigpara(1), 1, no_Layers);
    else
        sigscale = sigpara(1:no_Layers);
    end
    
    if length(sigpara1) < no_Layers
        sigscale1 = repmat(sigpara1(1), 1, no_Layers);
    else
        sigscale1 = sigpara1(1:no_Layers);
    end
    
    % Usar el vector C1 pasado como argumento
    if length(C1) < (no_Layers + 1)
        % Si C1 es más corto, expandir con el último valor
        C = [C1, repmat(C1(end), 1, no_Layers + 1 - length(C1))];
    else
        C = C1(1:(no_Layers + 1));
    end
    
    InputDataLayer = P;
    if Testcode == 1, rng('default'); end

    % ========== CAPAS OCULTAS ==========
    for i = 1:no_Layers
        % Pesos aleatorios ortonormalizados
        InputWeight = rand(HN(i+1), HN(i))*2 - 1;
        if HN(i+1) > HN(i)
            InputWeight = orth(InputWeight);
        else
            InputWeight = orth(InputWeight')';
        end
        
        BiasofHiddenNeurons = rand(HN(i+1),1)*2 - 1;
        BiasofHiddenNeurons = orth(BiasofHiddenNeurons);

        tempH = InputWeight*InputDataLayer + BiasofHiddenNeurons*ones(1, size(InputDataLayer,2));
        H = 1./(1 + exp(-sigscale1(i)*tempH));

        % ✅ CORRECCIÓN: Cálculo de pesos con regularización opcional
        if HN(i+1) == HN(i)
            % Caso especial: dimensiones iguales (usar Procrustes)
            [~, stack{i}.w, ~] = procrustNew(InputDataLayer', H');
        else
            % Caso general: usar regularización si C(i) > 0
            if C(i) == 0
                % Sin regularización
                stack{i}.w = pinv(H')*InputDataLayer';
            else
                % Con regularización L2
                stack{i}.w = (H*H' + C(i)*eye(HN(i+1))) \ (H*InputDataLayer');
            end
        end

        tempH = stack{i}.w*InputDataLayer;

        if HN(i+1) == HN(i)
            InputDataLayer = tempH;
        else
            InputDataLayer = 1./(1 + exp(-sigscale(i)*tempH));
        end
    end

    % ========== CAPA DE SALIDA ==========
    % Usar regularización si C(no_Layers+1) > 0
    if C(no_Layers+1) == 0
        stack{no_Layers+1}.w = pinv(InputDataLayer')*T';
    else
        stack{no_Layers+1}.w = (InputDataLayer*InputDataLayer' + C(no_Layers+1)*eye(size(InputDataLayer,1))) \ (InputDataLayer*T');
    end

    TrainingTime_m = toc(train_time);
    Y_M = (InputDataLayer'*stack{no_Layers+1}.w)';

    % ========== TESTEO ==========
    TV.P = entradas_testeo';
    test_time = tic;
    InputDataLayer = TV.P;
    
    for i = 1:no_Layers
        tempH_test = stack{i}.w*InputDataLayer;
        if HN(i+1) == HN(i)
            InputDataLayer = tempH_test;
        else
            InputDataLayer = 1./(1 + exp(-sigscale(i)*tempH_test));
        end
    end
    
    TY_M = (InputDataLayer'*stack{no_Layers+1}.w)';
    TestingTime_m = toc(test_time);

    % ========== MÉTRICAS EN KG ==========
    TV_T = T_real_test;
    T = T_real_train;
    
    if Elm_Type == REGRESSION
        trainMSE_m = mse(T_real_train - Y_M);
        testMSE_m  = mse(T_real_test  - TY_M);
        
        TrainingAccuracy_m = sqrt(trainMSE_m);
        TestingAccuracy_m  = sqrt(testMSE_m);
        
        % Coeficiente de correlación
        if std(T_real_train(:))>0 && std(Y_M(:))>0
            trainR_m = corr(T_real_train(:), Y_M(:), 'Rows','complete');
        end
        if std(T_real_test(:))>0 && std(TY_M(:))>0
            testR_m = corr(T_real_test(:), TY_M(:), 'Rows','complete');
        end
        
        % Coeficiente de determinación R²
        SS_res_tr = sum((T_real_train - Y_M).^2);
        SS_tot_tr = sum((T_real_train - mean(T_real_train)).^2);
        trainR2_m = 1 - SS_res_tr/max(SS_tot_tr, eps);
        
        SS_res_te = sum((T_real_test - TY_M).^2);
        SS_tot_te = sum((T_real_test - mean(T_real_test)).^2);
        testR2_m  = 1 - SS_res_te/max(SS_tot_te, eps);
        
    else % CLASSIFIER
        MissClassificationRate_Training = 0;
        MissClassificationRate_Testing  = 0;

        for i = 1:size(T, 2)
            [~, label_index_expected] = max(T(:, i));
            [~, label_index_actual]   = max(Y_M(:, i));
            if label_index_actual ~= label_index_expected
                MissClassificationRate_Training = MissClassificationRate_Training + 1;
            end
        end
        TrainingAccuracy_m = 1 - MissClassificationRate_Training / size(T, 2);
        
        for i = 1:size(TV.T, 2)
            [~, label_index_expected] = max(TV.T(:, i));
            [~, label_index_actual]   = max(TY_M(:, i));
            if label_index_actual ~= label_index_expected
                MissClassificationRate_Testing = MissClassificationRate_Testing + 1;
            end
        end
        TestingAccuracy_m = 1 - MissClassificationRate_Testing / size(TV.T, 2);
    end
end

