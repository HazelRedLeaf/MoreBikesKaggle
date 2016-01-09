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
function [MAE, dataTable, x_table] = main()
    % open CSV fine, preprocess and save to a table variable
    dataTable = load_training_data();
    
    % 3.280280  2.703160  1.940838   with numDocks
    % 3.280280  2.698309  1.942795   without numDocks
    
    % chosen features
    interests = {'numDocks', 'hour', 'weekday', 'weekhour', ...
        'windMaxSpeedms', 'windMeanSpeedms', 'windDirectiongrades', ...
        'temperatureC', 'relHumidityHR', 'airPressuremb', ...
        'bikes_3h_ago', 'full_profile_bikes', 'full_profile_3h_diff_bikes'};
    
%     interests = {'station','latitude','longitude','numDocks','weekhour','bikes_3h_ago', ...
%         'full_profile_3h_diff_bikes','full_profile_bikes', ...
%         'short_profile_3h_diff_bikes','short_profile_bikes'};
    
    number_of_features = 4;

    % rank features' correlation with 'bikes'
    feature_index = feature_selection(dataTable, interests)
    
    % produce a model using Least Squares
    [a_ls, y_estimate, MAE, y, X] = least_squares(dataTable, interests, feature_index, number_of_features);
    fprintf('a_ls = %d\n', a_ls);

    % produce a new, more accurate model without the outliers
    [a_ls, y_estimate, MAE, y] = ls_no_outliers(dataTable, interests, feature_index, number_of_features, y, y_estimate, X);
    fprintf('a_ls = %d\n', a_ls);
    
    % remove outliers
    %[~newTable] = remove_outliers(dataTable, y_estimate);
    % remove_outliers(dataTable, y_estimate);
    
    % produce a new, more accurate model without the outliers
    
    % load test data in a similar fashion to the training data
    dataTest = load_test_data();
    
    % predict 'bikes' in the test data
    bikes_predicted = predict_bikes(a_ls, dataTest, interests, feature_index, number_of_features);

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
function[a_ls, y_estimate, MAE, y, X]=least_squares(dataTable, interests, feature_index , number_of_features)
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
    x_matrix = [x, y];
    
    x_table = array2table(x_matrix,...
            'VariableNames',{'bikes_3h_ago' 'full_profile_bikes'...
            'full_profile_3h_diff_bikes' 'numDocks' 'bikes'});
 
    assignin('base', 'x_table', x_table);
    
    mdl = fitlm(x, y, 'linear')
    
    fcn = @(Xtr, Ytr, Xte) predict(...
        fitlm(Xtr,Ytr,'linear'), ...
        Xte)

    %  3.6939   3.6936  3.6940
    %  3.6952
    
    % perform cross-validation, and return average MSE across folds
    mse = crossval('mse', x, y, 'Predfun',fcn, 'kfold',10)

    % compute root mean squared error
    avrg_rmse = sqrt(mse)
    
    % add a column of 1-s on the left of x
    % use in the least squares formula
    X = [ones(size(x, 1), 1) x];
    
    % least squares vector
    % pinv(X) = inv(X' * X) * X'
    a_ls = pinv(X) * y;
    
    % get predicted values for the training data
    % with the model just computed
    y_estimate = a_ls(1) + x * a_ls(2:end);
    
%     figure;
%     hold on;
%     x = 1:42900;
%     size(y)
%     scatter(x, y);
%     scatter(x, y_estimate, 'g*');
%     hold off;
    
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
function[a_ls, y_estimate, MAE, y]= ls_no_outliers(dataTable, interests, feature_index , number_of_features, y, y_estimate, X)
    % for each chosen feature
    % remove NaN rows
    % do least squares prediction for bikes
    % calculate MAE for bikes based
    % on each feature  
    
    
    % 2.356618 2.185221 2.076342 1.940838
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

    %  3.6939   3.6936  3.6940 ---> with numDocks
    %  3.6952 ---> without numDocks
    %  T    MSE    ---> without outliers
    %  3   1.9879
    % 3.5  2.2252
    %  4   2.4408 
    % 4.5  2.6355
    %  5   2.7963
    % 5.5  2.9357
    %  6   3.0603
    % 6.5  3.1672
    
    % perform cross-validation, and return average MSE across folds
    mse = crossval('mse', x, y, 'Predfun',fcn, 'kfold',10)

    % compute root mean squared error
    avrg_rmse = sqrt(mse)
    
    % least squares vector
    % pinv(X) = inv(X' * X) * X'
    a_ls = pinv(X) * y;
    
    % get predicted values for the training data
    % with the model just computed
    y_estimate = X * a_ls;
    
    % calculate the mean absolute error between 'bikes' and the predicted
    % 'bikes'
    MAE = mean(abs(y - y_estimate));
    fprintf('MAE without outliers is %f\n', MAE)
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
function [bikes_predicted] = predict_bikes(a_ls, dataTest, interests, feature_index, number_of_features)
    % use the best 4 features to avoid overfitting
    used_features_index = feature_index(1:number_of_features)
    x = [];
    for i = used_features_index
        x = [x dataTest.(interests{i})];
    end
    
    % remove rows with NaN as above
    b = isnan(x(:,1));
    for i = 2:number_of_features
        b = b | isnan(x(:,i));
    end
    x(b,:) = [];
    
    % get predicted values for the test data
    % with the a_ls model
    bikes_predicted = round(a_ls(1) + x * a_ls(2:end));
    
    bikes_predicted(bikes_predicted < 0) = 0;
    size(bikes_predicted)
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