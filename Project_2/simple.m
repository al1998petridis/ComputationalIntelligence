%% Load Data
clear;
data = load('airfoil_self_noise.dat');

%% Separation into training-validation-control sets  
% Normalize input data(first 5 columns) in [0,1]
% from Matlab repo of elearning
[train_set, validation_set, test_set] = split_scale(data, 1);

%% 4 Models
TSKs = 4;
MFs = [2 3 2 3];
% 2 Singelton, 2 Polynomial
MFsType = ["constant" "constant" "linear" "linear"]; 
char_inputs = ["Frequency" "Angle of attack" "Chord length" "Free-stream velocity"...
    "Suction side displacement thickness"];
char_output = "Scaled sound pressure level, in decibels.";

for i = 1:TSKs
    %% Generate fuzzy inference system object from data
    % FIS options
    options = genfisOptions('GridPartition');
    options.InputMembershipFunctionType = 'gbellmf';
    options.NumMembershipFunctions = MFs(i);
    options.OutputMembershipFunctionType = MFsType(i); 
    % FIS model
%     TSK_model = genfis(train_set(:,1:end-1), train_set(:,end), options);
   TSK_model(i) = genfis(train_set(:,1:end-1), train_set(:,end), options);
%     % Plot MFs before training
%     figure;
%     for j = 1:length(TSK_model(i).input)
%         subplot(2,3,j);
%         plotmf(TSK_model(i), 'input', j);
%         xlabel(char_inputs(j));
%     end
%     str = strcat('Model ', num2str(i), ': MFs before training');
%     title(str);
    %% Train TSK fuzzy inference system using training data
    % Training options
%     train_options = anfisOptions('InitialFis', TSK_model, 'EpochNumber', 100);
    train_options = anfisOptions('InitialFis', TSK_model(i), 'EpochNumber', 100);
    train_options.ValidationData = validation_set;
    % Train model
    [finalFIS,trainError,~,bestFIS,chkError] = anfis(train_set,train_options);
%     [~,~,~,bestFIS,~] = anfis(train_set,train_options);

%     % Plot MFs after training
%     figure;
%     for j = 1:length(finalFIS.input)
%         subplot(2,3,j);
%         plotmf(finalFIS, 'input', j);
%         xlabel(char_inputs(j));
%     end
%     str = strcat('Model ', num2str(i), ': MFs after training');
%     sgtitle(str);
    % Plot MFs after training when validation error is minimum
    figure;
    for j = 1:length(bestFIS.input)
        subplot(2,3,j);
        plotmf(bestFIS, 'input', j);
        xlabel(char_inputs(j));
    end
    str = strcat('Model_', num2str(i), ': MFs after training when validation error is minimum');
    sgtitle(str);
    % Plot Learning Curve of model
    figure;
    plot([trainError chkError]);
    xlabel('Number of epochs (iterations)');
    ylabel('Error');
    legend('Training Error', 'Validation Error');
    str = strcat('Model_', num2str(i), ': Learning Curve');
    title(str);    
    %% Evaluate Fuzzy Inference System
    Y_out = evalfis(bestFIS, test_set(:,1:end-1));
    % Plot Prediction Error
    error = abs(Y_out - test_set(:,end));
    figure;
    plot(error);
    xlabel('Testing Data');
    ylabel(strcat('Error of ', " ", char_output));
    str = strcat('Model_', num2str(i), ': Prediction Error');
    title(str);
    % Initialize Metrics
    % 1. MSE = Mean Square Error, RMSE = Root Mean Square Error
    RMSE(i) = sqrt(mse(Y_out,test_set(:,end)));
    % 2. Evaluation function (Determination Factor)
    Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2); % R^2 = 1 - (SS_res/SS_tot)
    R2(i) = Rsq(Y_out,test_set(:,end));
    % 3. NMSE, NDEI
    NMSE(i) = 1 - R2(i);
    NDEI(i) = sqrt(NMSE(i));
    fprintf(strcat('TSK_model_',int2str(i)));
    fprintf('\n');
    fprintf('RMSE = %f\nNMSE = %f\nNDEI = %f\nR2 = %f\n', RMSE(i), NMSE(i), NDEI(i), R2(i));
    fprintf('\n');
end



