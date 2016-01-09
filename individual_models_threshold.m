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
function [MAE, baselineMae] = main(num)
    % open CSV fine, preprocess and save to a table variable
    dataTable = load_training_data(num);
    
    % chosen features
    interests = {'bikes_3h_ago', 'hour', 'weekhour', 'temperatureC', ...
        'weekday', 'full_profile_bikes', 'windMaxSpeedms', ...
        'windMeanSpeedms', 'relHumidityHR', 'airPressuremb', ...
        'full_profile_3h_diff_bikes'};
    
    feature_threshold = 0.2;
    
    % rank features' correlation with 'bikes'
    feature_index = feature_selection(dataTable, interests, feature_threshold);
    
    % produce a model using Least Squares
    [a_ls, y_estimate, MAE, y, X, baselineMae] = least_squares(dataTable, interests, feature_index, feature_threshold);
    fprintf('a_ls = %d\n', a_ls);
    
    % produce a new, more accurate model without the outliers
    [a_ls, y_estimate, MAE, y] = ls_no_outliers(dataTable, interests, feature_index, feature_threshold, y, y_estimate, X);
    fprintf('a_ls = %d\n', a_ls);
    
    % load test data in a similar fashion to the training data
    dataTest = load_test_data();
    
    % predict 'bikes' in the test data
    bikes_predicted = predict_bikes(a_ls, dataTest, interests, feature_index, feature_threshold);

    % write to the leaderboard submission file
    write_leaderboard(bikes_predicted);
end


% load training data from a CSV file, and preprocess
function [dataTable] = load_training_data(num) % num - number of training data file 201-275
    % if that file has already been created, use it without reloading
    if exist(sprintf('%d.csv', num), 'file')
        dataTable = readtable(sprintf('%d.csv', num));
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTable = importfile(sprintf('Train/station_%d_deploy.csv', num));
        writetable(dataTable, sprintf('%d.csv', num));
    end
end

% find the correlation between 'bikes' and all chosen features, and rank
% from most to least correlated - to use in the model
function[feature_index] = feature_selection(dataTable, interests, feature_threshold)

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
    correlations = correlations(correlations >= feature_threshold);
    [~, feature_index] = sort(correlations, 'descend');

end

% use the features ranked and 'bikes' to compute least squares and predict 
% estimated 'bikes'; calculate MAE between 'bikes' and 'bikes'-estimate
function[a_ls, y_estimate, MAE, y, X, baselineMae]=least_squares(dataTable, interests, feature_index, feature_threshold)
    % for each chosen feature
    % remove NaN rows
    % do least squares prediction for bikes
    % calculate MAE for bikes_3h_ago based
    % on each feature
    
    % compose x - the feature matrix
    x = [];
    for i = feature_index
        x = [x dataTable.(interests{i})];
    end
    
    % y - the vector with bikes
    y = dataTable.bikes;
    
    y_baseline = dataTable.bikes_3h_ago;
 
    % remove rows with NaN as above
    b = isnan(y);
    for i = 1:size(feature_index,2)
        b = b | isnan(x(:,i));
    end
    x(b,:) = [];
    y(b) = [];
    y_baseline(b) = [];
    
    % add a column of 1-s on the left of x
    % use in the least squares formula
    X = [ones(size(x, 1), 1) x];
    
    % least squares vector
    % pinv(X) = inv(X' * X) * X'
    a_ls = pinv(X) * y;
    
    % get predicted values for the training data
    % with the model just computed
    y_estimate = X * a_ls;

    % calculate the mean absolute error between 'bikes' and the predicted
    % 'bikes'
    MAE = mean(abs(y - y_estimate));
    baselineMae = mean(abs(y - y_baseline))
    fprintf('Mean absolute error is %f\n', MAE)
end

function[a_ls, y_estimate, MAE, y, X]= ls_no_outliers(dataTable, interests, feature_index , number_of_features, y, y_estimate, X)
    % for each chosen feature
    % remove NaN rows
    % do least squares prediction for bikes
    % calculate MAE for bikes based
    % on each feature  
    
    T = 2.3;
    
    % remove outliers
    bikes_comp = [y  y_estimate];
    % calculate standard deviation of y and y_estimate
    bikes_std = std(bikes_comp');
    y(bikes_std >= T) = [];
    y_estimate(bikes_std >= T) = [];
    X(bikes_std >= T,:) = [];
   
    close all;
    figure;
    hold on;
    scatter(y,y_estimate);
    hold off;
    
    % least squares vector
    % pinv(X) = inv(X' * X) * X'
    a_ls = pinv(X) * y;
    
    % get predicted values for the training data
    % with the model just computed
    y_estimate = X * a_ls;
    
    % calculate the mean absolute error between 'bikes' and the predicted
    % 'bikes'
    sprintf('HERE')
    MAE = mean(abs(y - y_estimate));
    fprintf('Mean absolute error is %f\n', MAE)
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
function [bikes_predicted] = predict_bikes(a_ls, dataTest, interests, feature_index, feature_threshold)
    % use the best features above threshold to avoid overfitting
    x = [];
    for i = feature_index
        x = [x dataTest.(interests{i})];
    end
    
    % remove rows with NaN as above
    b = isnan(x(:,1));
    for i = 2:size(feature_index)
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
    %writetable(dataLeaderboard, sprintf('leaderboard_submission_%d.csv', num));

    dataLeaderboard.bikes = bikes;
    writetable(dataLeaderboard, sprintf('leaderboard_submission_%d.csv', num));
end