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
function [MAE] = main(num, num2)
    % open CSV fine, preprocess and save to a table variable
    modelTable = load_model(num);
    % dataTable = load_test_data();
    
    dataTable = load_training_data(num2);
    
    % chosen features
    a_ls = modelTable.weight
    interests = modelTable.feature
    interests_size = size(interests,1);
    
    x = [];
    for i = 2:interests_size
        x = [x interests(i)];
    end
    
    X = [];
    for x = 2:4
        X = [X dataTable.(interests{x})];
    end
    
    X = [X dataTable.temperatureC];
    
     size(X)
    
    % y - the vector with bikes
    y = dataTable.bikes;
    size(y)
    
    % remove rows with NaN as above
    b = isnan(y);
    for i = 1:3
        b = b | isnan(X(:,i));
    end
    X(b,:) = [];
    y(b) = [];
    
    X = [ones(size(X, 1), 1) X];
    a_ls;
    bikes = round(X*a_ls);
    bikes(bikes < 0) = 0;
    
     MAE = mean(abs(y - bikes));
    fprintf('Mean absolute error is %f\n', MAE)

    sprintf('MOO')
    
    % predict 'bikes' in the test data
    %bikes_predicted = predict_bikes(a_ls, dataTest, interests, feature_index, number_of_features);

    % write to the leaderboard submission file
    % write_leaderboard(bikes);
end


% load training data from a CSV file, and preprocess
function [dataTable] = load_model(num) % num - number of model files 1 - 200
    % if that file has already been created, use it without reloading
    if exist(sprintf('model_station_%d_rlm_short_temp.csv', num), 'file')
        dataTable = readtable(sprintf('model_station_%d_rlm_short_temp.csv', num));
        dataTable
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTable = importmodel(sprintf('Models/model_station_%d_rlm_short_temp.csv', num));
        writetable(dataTable, sprintf('model_station_%d_rlm_short_temp.csv', num));
        dataTable
    end
    
end

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