% Kaggle Bikes 2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% all features:                                                           %
% station, latitude, longitude, numDocks, timestamp, year, month, day,    %
% hour, weekday, weekhour, isHoliday, windMaxSpeedms, windMeanSpeedms,    %
% windDirectiongrades, temperatureC, relHumidityHR, airPressuremb,        %
% precipitationlm2, bikes_3h_ago, full_profile_3h_diff_bikes,             %
% full_profile_bikes, short_profile_3h_diff_bikes, short_profile_bikes,   %
% bikes                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% main function
function [MAE3] = main()
    % open CSV fine, preprocess and save to a table variable
    dataTable = load_training_data();
        
    % chosen features
    interests = {'numDocks', 'hour', 'weekday', 'weekhour', ...
        'windMaxSpeedms', 'windMeanSpeedms', 'windDirectiongrades', ...
        'temperatureC', 'relHumidityHR', 'airPressuremb', ...
        'bikes_3h_ago', 'full_profile_bikes', 'full_profile_3h_diff_bikes'};
    
    number_of_features = 4;

    % rank features' correlation with 'bikes'
    feature_index = feature_selection(dataTable, interests)
    
    % produce a model using Least Squares
    [a_ls_1, y_estimate, MAE, y, X] = least_squares(dataTable, interests, feature_index, number_of_features);
    fprintf('Phase 1: With outliers: a_ls = %d\n', a_ls_1);

    % produce a new, more accurate model without the outliers
    [a_ls_1, y_estimate, MAE, y] = ls_no_outliers(dataTable, interests, feature_index, number_of_features, y, y_estimate, X);
    fprintf('Phase 1: Without outliers: a_ls = %d\n', a_ls_1);
    
    
    general_model_features = {'bikes_3h_ago', 'full_profile_bikes', ...
        'full_profile_3h_diff_bikes', 'numDocks'};
    
    short_model_features = {'bikes_3h_ago', 'full_profile_bikes',...
        'full_profile_3h_diff_bikes', 'temperatureC', ...
        'short_profile_bikes', 'short_profile_3h_diff_bikes'};
    
    all_features = {'bikes_3h_ago', ...
        'full_profile_bikes', 'full_profile_3h_diff_bikes'};
    
    [a_ls_2, MAE_short, baselineMAEvec, station] = profile_short();
    fprintf('Phase 2: a_ls = %d\n', a_ls_2);
    fprintf('Phase 2: MAE', MAE_short);
    
    intercept = (a_ls_1(1) + 12*a_ls_2(1)) / 13;
    bikes3hago = (a_ls_1(2) + 12*a_ls_2(2)) / 13;
    fullprofilebikes = (a_ls_1(3) + 12*a_ls_2(3)) / 13;
    fullprofile3hdiffbikes = (a_ls_1(4) + 12*a_ls_2(4)) / 13;
    %numDocks = a_ls_1(5)/13;
    %temperatureC = 12*a_ls_2(5)/13;
    %shortprofilebikes = 12*a_ls_2(6)/13;
    %shortprofile3hdiffbikes = 12*a_ls_2(7)/13;
    
    % all features
    a_ls = [intercept; bikes3hago; fullprofilebikes; fullprofile3hdiffbikes; ...
        ]
        
    fprintf('Phase 3: a_ls = %d\n', a_ls);
    
    X3 = [];
    for feature = all_features
        X3 = [X3 dataTable.(feature{1})];
    end
   
    y3 = dataTable.bikes;
    
    % remove rows with NaN as above
    b = isnan(y3);
    for i = 1:3
        b = b | isnan(X3(:,i));
    end
    X3(b,:) = [];
    y3(b) = [];

    % get predicted values for the training data
    % with the model just computed
    y_estimate3 = a_ls(1) + X3 * a_ls(2:end);
    
    % calculate the mean absolute error between 'bikes' and the predicted
    % 'bikes'
    MAE3 = mean(abs(y3 - y_estimate3));
    fprintf('Mean absolute error BEFORE TEST DATA is %f\n', MAE3)
    
    % load test data in a similar fashion to the training data
    dataTest = load_test_data();
     
    % predict 'bikes' in the test data
    bikes_predicted = predict_bikes(a_ls, dataTest, all_features);
 
    % write to the leaderboard submission file
    write_leaderboard(bikes_predicted);
end


% load training data from a CSV file, and preprocess
function [dataTable] = load_training_data() % num - number of training data file 201-275
    % if that file has already been created, use it without reloading
    dataTable = [];
    for num = 201:275
        if exist(sprintf('%d.csv', num), 'file')
            dataTable = [dataTable; readtable(sprintf('%d.csv', num))];
        else
            % load the file into a table
            % changes weekdays to numbers; and NA to NaN
            dataTable = importfile(sprintf('Train/station_%d_deploy.csv', num));
            writetable(dataTable, sprintf('%d.csv', num));
        end
    end
end

% find the correlation between 'bikes' and all chosen features, and rank
% from most to least correlated - to use in the model
function[feature_index] = feature_selection(dataTable, interests)

    % 201 results: bikes_3h_ago, full_profile_bikes, weekhour, weekday, temperatureC, hour; 
    % 202 results: bikes_3h_ago, full_profile_bikes, hour, weekday, weekhour, temperatureC
    
    % empty array for the corr coefs
    correlations = [];
    % for each of the chosen features
    for interest = interests
        
        % x - vector with the current feature
        x = dataTable.(interest{1});
        
        % y - bikes
        y = dataTable.bikes;
        
        % remove rows with NaN (rather than substituting with 0s)
        % b = 1 if either x or y is Nan (bitwise OR)
        b = isnan(x) | isnan(y);
        % for all b = 1, delete this row in both x and y
        x(b) = [];
        y(b) = [];
        
        % find the correlation coefficient of 'bikes' and the current
        % feature
        correlation = corrcoef(x, y);
        % result is a matrix of the type:
        %        1.000 corrcoef
        %        corrcoef 1.000
        % so get the corrcoef value only
        correlation = correlation(2,1);
        
        % add the 'current' correlation's absolute value to the previous
        % ones
        correlations = [correlations abs(correlation)];
        
        fprintf('%s = %f\n', interest{1}, correlation);
    end

    % sort correlations in a descending order
    [~, feature_index] = sort(correlations, 'descend');
end

% use the features ranked and 'bikes' to compute least squares and predict 
% estimated 'bikes'; calculate MAE between 'bikes' and 'bikes'-estimate
function[a_ls_1, y_estimate, MAE, y, X]=least_squares(dataTable, interests, feature_index , number_of_features)
    % for each chosen feature
    % remove NaN rows
    % do least squares prediction for bikes
    % calculate MAE for bikes based
    % on each feature
  
    % use the best number of features to avoid overfitting
    used_features_index = feature_index(1:number_of_features)
    
    % compose x - the feature matrix
    x = [];
    for i = used_features_index
        x = [x dataTable.(interests{i})];
    end
    
    % y - the vector with bikes
    y = dataTable.bikes;
    
    y_baseline = dataTable.bikes_3h_ago;
    y_baseline2 = dataTable.bikes_3h_ago + dataTable.full_profile_3h_diff_bikes;
 
    % remove rows with NaN as above
    b = isnan(y);
    for i = 1:number_of_features
        b = b | isnan(x(:,i));
    end
    x(b,:) = [];
    y(b) = [];
    y_baseline(b) = [];
    y_baseline2(b) = [];
    
    mdl = fitlm(x, y, 'linear')
    
    fcn = @(Xtr, Ytr, Xte) predict(...
        fitlm(Xtr,Ytr,'linear'), ...
        Xte)
    
    % perform cross-validation, and return average MSE across folds
    mse = crossval('mse', x, y, 'Predfun',fcn, 'kfold',10)

    % compute root mean squared error
    avrg_rmse = sqrt(mse)
    
    % add a column of 1-s on the left of x
    % use in the least squares formula
    X = [ones(size(x, 1), 1) x];
    
    % least squares vector
    a_ls_1 = pinv(X) * y;
    
    % get predicted values for the training data
    % with the model just computed
    y_estimate = a_ls_1(1) + x * a_ls_1(2:end);
 
    % calculate the mean absolute error between 'bikes' and the predicted
    % 'bikes'
    MAE = mean(abs(y - y_estimate));
    baselineMAE = mean(abs(y - y_baseline));
    baselineMAE2 = mean(abs(y - y_baseline2));
    fprintf('Mean absolute error at first is %f\n', MAE)
    fprintf('Baseline MAE is %f\n', baselineMAE)
    fprintf('Baseline2 MAE is %f\n', baselineMAE2)
end

% use the features ranked and 'bikes' to compute least squares and predict 
% estimated 'bikes'; calculate MAE between 'bikes' and 'bikes'-estimate
function[a_ls_1, y_estimate, MAE, y]= ls_no_outliers(dataTable, interests, feature_index , number_of_features, y, y_estimate, X)
    % for each chosen feature
    % remove NaN rows
    % do least squares prediction for bikes
    % calculate MAE for bikes based
    % on each feature  
    
    T = 4;
    
    % remove outliers
    bikes_comp = [y  y_estimate];
    % calculate standard deviation of y and y_estimate
    bikes_std = std(bikes_comp');
    y(bikes_std >= T) = [];
    y_estimate(bikes_std >= T) = [];
    X(bikes_std >= T,:) = [];
   
    x = [X(:,2) X(:,3) X(:,4) X(:,5)];
    
    close all;
    figure;
    axis square;
    hold on;
    scatter(y,y_estimate,'m');
    hold off;
    
    mdl = fitlm(x, y, 'linear')
    
    fcn = @(Xtr, Ytr, Xte) predict(...
        fitlm(Xtr,Ytr,'linear'), ...
        Xte)
    
    % perform cross-validation, and return average MSE across folds
    mse = crossval('mse', x, y, 'Predfun',fcn, 'kfold',10)

    % compute root mean squared error
    avrg_rmse = sqrt(mse)
    
    % least squares vector
    % pinv(X) = inv(X' * X) * X'
    a_ls_1 = pinv(X) * y;
    
    % get predicted values for the training data
    % with the model just computed
    y_estimate = X * a_ls_1;
    
    % calculate the mean absolute error between 'bikes' and the predicted
    % 'bikes'
    MAE = mean(abs(y - y_estimate));
    fprintf('MAE without outliers is %f\n', MAE)
end

function[a_ls_2, MAE_short, baselineMAEvec, station] = profile_short()
    station = [];
    MAE_short = [];
    baselineMAEvec = [];
    %for station_num = 201:275
    a_ls_2 = [];
    intercept_tot = 0;
    bikes3hago_tot = 0;
    fullprofilebikes_tot = 0;
    fullprofile3hdiffbikes_tot = 0;
    temp_tot = 0;
    shortprofilebikes_tot = 0;
    shortprofile3hdiffbikes_tot = 0;

    for model_num = 1:200
        %fprintf('MAE_short | station: %d, model: %d\n', station_num, model_num);
            
        % 4 profiles
        dataModelShortFullTemp = load_model_short_full_temp(model_num);
        dataModelShortFull = load_model_short_full(model_num);
        dataModelShortTemp = load_model_short_temp(model_num);
        dataModelShort = load_model_short(model_num);
            
        %dataTable = load_training_data(station_num);

        % construct a_ls coefficients
        interceptMean = (dataModelShortFullTemp.weight(1) + ...
            dataModelShortFull.weight(1) + ...
            dataModelShortTemp.weight(1) + ...
            dataModelShort.weight(1))/4;                

        bikes3hagoMean = (dataModelShortFullTemp.weight(2) + ...
            dataModelShortFull.weight(2) + ...
            dataModelShortTemp.weight(2) + ...
            dataModelShort.weight(2))/4;           

        shortprofilebikesMean = (dataModelShortFullTemp.weight(3) + ...
            dataModelShortFull.weight(3) + ...
            dataModelShortTemp.weight(3) + ...
            dataModelShort.weight(3))/4;  

        shortprofile3hdiffbikesMean = (dataModelShortFullTemp.weight(4) + ...
            dataModelShortFull.weight(4) + ...
            dataModelShortTemp.weight(4) + ...
            dataModelShort.weight(4))/4;  

        fullprofilebikesMean = (dataModelShortFullTemp.weight(5) + ...
            dataModelShortFull.weight(5))/2;  

        fullprofile3hdiffbikesMean = (dataModelShortFullTemp.weight(6) + ...
            dataModelShortFull.weight(6))/2;
            
        tempMean = (dataModelShortFullTemp.weight(7) + ...
            dataModelShortTemp.weight(5))/2;

        intercept_tot = intercept_tot + interceptMean;
        bikes3hago_tot = bikes3hago_tot + bikes3hagoMean;
        fullprofilebikes_tot = fullprofilebikes_tot + fullprofilebikesMean;
        fullprofile3hdiffbikes_tot = fullprofile3hdiffbikes_tot + fullprofile3hdiffbikesMean;
        temp_tot = temp_tot + tempMean;
        shortprofilebikes_tot = shortprofilebikes_tot + shortprofilebikesMean;
        shortprofile3hdiffbikes_tot = shortprofile3hdiffbikes_tot + shortprofile3hdiffbikesMean;
    end

    a_ls_2 = [intercept_tot/200 bikes3hago_tot/200 fullprofilebikes_tot/200 fullprofile3hdiffbikes_tot/200 temp_tot/200 shortprofilebikes_tot/200 shortprofile3hdiffbikes_tot/200]'

        % Retrieve MAE and baseline MAE from estimate LS function
%         [MAE, baselineMAE] = leastSquares(a_ls, dataTable);
%         
%         station = [station; station_num];
%         
%         baselineMAEvec = [baselineMAE; baselineMAEvec]
%         MAE_short = [MAE_short; MAE];       
%         
        % load test data in a similar fashion to the training data
        %dataTest = load_test_data();
        %interests = {'bikes_3h_ago', 'full_profile_bikes', 'full_profile_3h_diff_bikes', 'temperatureC', 'short_profile_bikes', 'short_profile_3h_diff_bikes'};
       % bikes_predicted = predict_bikes(a_ls, dataTest, interests)
        % write to the leaderboard submission file
       % write_leaderboard(bikes_predicted);
    %end
end

% load test data from a CSV file, and preprocess
function [dataTest] = load_test_data() % num - number of training data file 201-275
    % if that file has already been created, use it without reloading
    if exist(sprintf('testLoaded.csv'), 'file')
        dataTest = readtable(sprintf('testLoaded.csv'));
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTest = importtest(sprintf('test.csv'));
        writetable(dataTest, sprintf('testLoaded.csv'));
    end
end

% predict 'bikes' in the test data
function [bikes_predicted] = predict_bikes(a_ls, dataTest, interests)
    % use the best 4 features to avoid overfitting
    x = [];
    for interest = interests
        x = [x dataTest.(interest{1})];
    end
    
    % remove rows with NaN as above
    b = isnan(x(:,1));
    for i = 2:size(interests)
        b = b | isnan(x(:,i));
    end
    x(b,:) = [];
    
    % get predicted values for the test data
    % with the a_ls model
    bikes_predicted = round(a_ls(1) + x * a_ls(2:end));
    
    bikes_predicted(bikes_predicted < 0) = 0;
    size(bikes_predicted);
end

% write to the leaderboard submission file
function[] = write_leaderboard(bikes)
    % avoid overwriting of leaderboard_submission files
    num = 1;
    while exist(sprintf('leaderboard_submission_%d.csv', num), 'file')
        num = num + 1;
    end
    
    dataLeaderboard = importleaderboard(sprintf('example_leaderboard_submission.csv'));
    dataLeaderboard.bikes = bikes;
    writetable(dataLeaderboard, sprintf('leaderboard_submission_%d.csv', num));
end

function [dataTable] = load_model_short_full_temp(num) % num - number of model files 1 - 200
    % if that file has already been created, use it without reloading
    if exist(sprintf('model_station_%d_rlm_short_full_temp.csv', num), 'file')
        dataTable = readtable(sprintf('model_station_%d_rlm_short_full_temp.csv', num));
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTable = importmodel(sprintf('Models/model_station_%d_rlm_short_full_temp.csv', num));
        writetable(dataTable, sprintf('model_station_%d_rlm_short_full_temp.csv', num));
    end
end

function [dataTable] = load_model_short_full(num) % num - number of model files 1 - 200
    % if that file has already been created, use it without reloading
    if exist(sprintf('model_station_%d_rlm_short_full.csv', num), 'file')
        dataTable = readtable(sprintf('model_station_%d_rlm_short_full.csv', num));
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTable = importmodel(sprintf('Models/model_station_%d_rlm_short_full.csv', num));
        writetable(dataTable, sprintf('model_station_%d_rlm_short_full.csv', num));
    end
end

function [dataTable] = load_model_short_temp(num) % num - number of model files 1 - 200
    % if that file has already been created, use it without reloading
    if exist(sprintf('model_station_%d_rlm_short_temp.csv', num), 'file')
        dataTable = readtable(sprintf('model_station_%d_rlm_short_temp.csv', num));
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTable = importmodel(sprintf('Models/model_station_%d_rlm_short_temp.csv', num));
        writetable(dataTable, sprintf('model_station_%d_rlm_short_temp.csv', num));
    end
end

function [dataTable] = load_model_short(num) % num - number of model files 1 - 200
    % if that file has already been created, use it without reloading
    if exist(sprintf('model_station_%d_rlm_short.csv', num), 'file')
        dataTable = readtable(sprintf('model_station_%d_rlm_short.csv', num));
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTable = importmodel(sprintf('Models/model_station_%d_rlm_short.csv', num));
        writetable(dataTable, sprintf('model_station_%d_rlm_short.csv', num));
    end
end