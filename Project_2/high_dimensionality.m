%% Load data
clear;
% data = readmatrix('train.csv');        % For matlab > 2019
data = dlmread('train.csv', ',', 1,0);              % For matlab < 2019

%% Separation into training-validation-control sets  
% Normalize input data(first 5 columns) in [0,1]
% from Matlab repo of elearning
[train_set, validation_set, test_set] = split_scale(data, 1);

%% Grid Partitioning
numberOfFeatures = [5, 10, 15, 20, 30];
ra = [0.2, 0.3, 0.4, 0.5, 0.8];
% 5-fold Cross Validation 
k = 5;
c = cvpartition(size(train_set,2), 'KFold', k); 
% Rank importance of predictors using RReliefF algorithm
[idx,weights] = relieff(train_set(:,1:end-1),train_set(:,end),10);  % kNN with k = 10
% Grid Search
grid_table = zeros(length(numberOfFeatures), length(ra));
for i = 1:length(numberOfFeatures)
    for j = 1:length(ra)
        err = zeros(k, 1);
        for l = 1:k
            % Train and test sets
            train = train_set(c.training(l) == 1, [idx(1:numberOfFeatures(i)), end]);
            test = train_set(c.test(l) == 1, [idx(1:numberOfFeatures(i)), end]);
            % FIS options
            options = genfisOptions('SubstractiveClustering', 'ClusterInfluenceRange', ra(j));
            % FIS model
            model = genfis(train(:,1:end-1), train(:,end), options);
            % Train model
            % Training options
            train_options = anfisOptions('InitialFis', model, 'EpochNumber', 20);
            train_options.ValidationData = test;
            % Train model
            [~,~,~,~,chkError] = anfis(train,train_options);            
            err(l) = min(chkError);
        end
        grid_table(i,j) = sum(err)/k;
    end
end
%% Parametric Plots for error
figure;
bar3(grid_table);
ylabel('Number of Features');
yticklabels({'5', '10', '15', '20', '30'});
xlabel('Radius Values');
xticklabels({'0.2', '0.3', '0.4', '0.5', '0.8'});
zlabel('Mean Square Error');
title('Error for different number of features and radius values 3D');     
    
%% Best model
[feature, radius] = find(grid_table == min(grid_table(:)));
attr = idx(1:numberOfFeatures(feature));
% FIS options
b_options = genfisOptions('SubstractiveClustering', 'ClusterInfluenceRange', ra(radius));
% FIS model
Best_model = genfis(train_set(:,attr), train_set(:,end), b_options);
% Training options
b_train_options = anfisOptions('InitialFis', Best_model, 'EpochNumber', 100);
b_train_options.ValidationData = validation_set(:,[attr, end]);
% Train model
[finalFIS,trainError,~,bestFIS,chkError] = anfis(train_set(:,[attr,end]),b_train_options);

%% Evaluate Fuzzy Inference System
Y_out = evalfis(bestFIS, test_set(:,attr));
% Initialize Metrics
% 1. MSE = Mean Square Error, RMSE = Root Mean Square Error
RMSE = sqrt(mse(Y_out,test_set(:,end)));
% 2. Evaluation function (Determination Factor)
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2); % R^2 = 1 - (SS_res/SS_tot)
R2 = Rsq(Y_out,test_set(:,end));
% 3. NMSE, NDEI
NMSE = 1 - R2;
NDEI = sqrt(NMSE);
fprintf(strcat('BEST_model Metrics'));
fprintf('\n');
fprintf('RMSE = %f\nNMSE = %f\nNDEI = %f\nR2 = %f\n', RMSE, NMSE, NDEI, R2);
fprintf('\n');

%% Plots
% Plot Predicted and Real Values
figure;
plot([test_set(:,end) Y_out]);
legend('Real', 'Predicted');
title('Predicted and Real Values');
% Plot Learning Curve of model
figure;
plot([trainError chkError]);
xlabel('Number of epochs (iterations)');
ylabel('Error');
legend('Training Error', 'Validation Error');
title('Best Model: Learning Curve');  
% Plot some MFs before training 
figure;
for i = 1:3
    subplot(2,3,i);
    plotmf(Best_model, 'input', i);
    title(strcat(int2str(i),'MF before training'));
end
% Plot some MFs after training 
for i = 1:3
    subplot(2,3,i+3);
    plotmf(bestFIS, 'input', i);
    title(strcat(int2str(i),'MF after training'));
end



            
            
            
