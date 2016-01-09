function [MAE, baselineMAE] = main()
    %[MAE_all, baselineMAEvec, station] = profile_all();
    %[MAE_full, baselineMAEvec, station] = profile_full();
    [MAE_short, baselineMAEvec, station] = profile_short();
    %[MAE_temp, baselineMAEvec, station] = profile_temp();
    %writetable(table(station,baselineMAEvec,MAE_all,MAE_full,MAE_short,MAE_temp),'phaseTwoProfileMAEs.csv');
    fprintf('Finished!\n');
end

function[MAE_all, baselineMAEvec, station] = profile_all()
    station = [];
    MAE_all = [];
    baselineMAEvec = [];
    for station_num = 201:275
        a_ls = [];
        intercept_tot = 0;
        bikes3hago_tot = 0;
        fullprofilebikes_tot = 0;
        fullprofile3hdiffbikes_tot = 0;
        temp_tot = 0;
        shortprofilebikes_tot = 0;
        shortprofile3hdiffbikes_tot = 0;

        for model_num = 1:200
        fprintf('MAE_all | station: %d, model: %d\n', station_num, model_num);
            
            % 6 profiles
            dataModelFull = load_model_full(model_num);
            dataModelFullTemp = load_model_full_temp(model_num);
            dataModelShortFullTemp = load_model_short_full_temp(model_num);
            dataModelShortFull = load_model_short_full(model_num);
            dataModelShortTemp = load_model_short_temp(model_num);
            dataModelShort = load_model_short(model_num);
            
            dataTable = load_training_data(station_num);

            % construct a_ls coefficients
            interceptMean = (dataModelFull.weight(1) + ...
                dataModelFullTemp.weight(1) + ...
                dataModelShortFullTemp.weight(1) + ...
                dataModelShortFull.weight(1) + ...
                dataModelShortTemp.weight(1) + ...
                dataModelShort.weight(1))/6;
            
            bikes3hagoMean = (dataModelFull.weight(2) + ...
                dataModelFullTemp.weight(2) + ...
                dataModelShortFullTemp.weight(2) + ...
                dataModelShortFull.weight(2) + ...
                dataModelShortTemp.weight(2) + ...
                dataModelShort.weight(2))/6;
            
            shortprofilebikesMean = (dataModelShortFullTemp.weight(3) + ...
                dataModelShortFull.weight(3) + ...
                dataModelShortTemp.weight(3) + ...
                dataModelShort.weight(3))/4;
          
            shortprofile3hdiffbikesMean = (dataModelShortFullTemp.weight(4) + ...
                dataModelShortFull.weight(4) + ...
                dataModelShortTemp.weight(4) + ...
                dataModelShort.weight(4))/4;
            
            fullprofilebikesMean = (dataModelFull.weight(3) + ...
                dataModelFullTemp.weight(3) + ...
                dataModelShortFullTemp.weight(5) + ...
                dataModelShortFull.weight(5))/4;
           
            fullprofile3hdiffbikesMean = (dataModelFull.weight(4) + ...
                dataModelFullTemp.weight(4) + ...
                dataModelShortFullTemp.weight(6) + ...
                dataModelShortFull.weight(6))/4;
                                  
            tempMean = (dataModelFullTemp.weight(5) + ...
                dataModelShortFullTemp.weight(7) + ...
                dataModelShortTemp.weight(5))/3;
            
            intercept_tot = intercept_tot + interceptMean;
            bikes3hago_tot = bikes3hago_tot + bikes3hagoMean;
            fullprofilebikes_tot = fullprofilebikes_tot + fullprofilebikesMean;
            fullprofile3hdiffbikes_tot = fullprofile3hdiffbikes_tot + fullprofile3hdiffbikesMean;
            temp_tot = temp_tot + tempMean;
            shortprofilebikes_tot = shortprofilebikes_tot + shortprofilebikesMean;
            shortprofile3hdiffbikes_tot = shortprofile3hdiffbikes_tot + shortprofile3hdiffbikesMean;
        end

        a_ls = [intercept_tot/200 bikes3hago_tot/200 fullprofilebikes_tot/200 fullprofile3hdiffbikes_tot/200 temp_tot/200 shortprofilebikes_tot/200 shortprofile3hdiffbikes_tot/200]'

        % Retrieve MAE and baseline MAE from estimate LS function
        [MAE, baselineMAE] = leastSquares(a_ls, dataTable);
        
        station = [station; station_num];
        
        baselineMAEvec = [baselineMAE; baselineMAEvec]
        MAE_all = [MAE_all; MAE];       
        
        % load test data in a similar fashion to the training data
        % dataTest = load_test_data()
        % bikes_predicted = round(X*a_ls)
        % bikes_predicted(bikes_predicted < 0) = 0;
        %size(bikes_predicted);
        % write to the leaderboard submission file
        %write_leaderboard(bikes_predicted);
    end
end

function[MAE_full, baselineMAEvec, station] = profile_full()
    station = [];
    MAE_full = [];
    baselineMAEvec = [];
    %for station_num = 201:275
        intercept_tot = 0;
        bikes3hago_tot = 0;
        fullprofilebikes_tot = 0;
        fullprofile3hdiffbikes_tot = 0;
        temp_tot = 0;
        shortprofilebikes_tot = 0;
        shortprofile3hdiffbikes_tot = 0;

        for model_num = 1:200
            %fprintf('MAE_full | station: %d, model: %d\n', station_num, model_num);
            
            % 4 profiles
            dataModelFull = load_model_full(model_num);
            dataModelFullTemp = load_model_full_temp(model_num);
            dataModelShortFullTemp = load_model_short_full_temp(model_num);
            dataModelShortFull = load_model_short_full(model_num);
            
            
            % construct a_ls coefficients
            interceptMean = (dataModelFull.weight(1) + ...
                dataModelFullTemp.weight(1) + ...
                dataModelShortFullTemp.weight(1) + ...
                dataModelShortFull.weight(1))/4;
            
            bikes3hagoMean = (dataModelFull.weight(2) + ...
                dataModelFullTemp.weight(2) + ...
                dataModelShortFullTemp.weight(2) + ...
                dataModelShortFull.weight(2))/4;
                       
            shortprofilebikesMean = (dataModelShortFullTemp.weight(3) + ...
                dataModelShortFull.weight(3))/2;
            
            shortprofile3hdiffbikesMean = (dataModelShortFullTemp.weight(4) + ...
                dataModelShortFull.weight(4))/2;
            
            fullprofilebikesMean = (dataModelFull.weight(3) + ...
                dataModelFullTemp.weight(3) + ...
                dataModelShortFullTemp.weight(5) + ...
                dataModelShortFull.weight(5))/4;
            
            fullprofile3hdiffbikesMean = (dataModelFull.weight(4) + ...
                dataModelFullTemp.weight(4) + ...
                dataModelShortFullTemp.weight(6) + ...
                dataModelShortFull.weight(6))/4;
            
            tempMean = (dataModelFullTemp.weight(5) + ...
                dataModelShortFullTemp.weight(7))/2;

            intercept_tot = intercept_tot + interceptMean;
            bikes3hago_tot = bikes3hago_tot + bikes3hagoMean;
            fullprofilebikes_tot = fullprofilebikes_tot + fullprofilebikesMean;
            fullprofile3hdiffbikes_tot = fullprofile3hdiffbikes_tot + fullprofile3hdiffbikesMean;
            shortprofilebikes_tot = shortprofilebikes_tot + shortprofilebikesMean;
            shortprofile3hdiffbikes_tot = shortprofile3hdiffbikes_tot + shortprofile3hdiffbikesMean;
            temp_tot = temp_tot + tempMean;

        end

        a_ls = [intercept_tot/200 bikes3hago_tot/200 fullprofilebikes_tot/200 fullprofile3hdiffbikes_tot/200 temp_tot/200 shortprofilebikes_tot/200 shortprofile3hdiffbikes_tot/200]'

        % dataTable = load_training_data(station_num);
        % Retrieve MAE and baseline MAE from estimate LS function
        % [MAE, baselineMAE] = leastSquares(a_ls, dataTable);
        
        %station = [station; station_num];
        
        %baselineMAEvec = [baselineMAE; baselineMAEvec]
        %MAE_full = [MAE_full; MAE];       
        
        % load test data in a similar fashion to the training data
        dataTest = load_test_data();
        interests = {'bikes_3h_ago', 'short_profile_bikes', 'short_profile_3h_diff_bikes', 'full_profile_bikes', 'full_profile_3h_diff_bikes', 'temperatureC'};
        bikes_predicted = predict_bikes(a_ls, dataTest, interests);
        % bikes_predicted = round(X*a_ls)
        %bikes_predicted(bikes_predicted < 0) = 0;
        %size(bikes_predicted);
        % write to the leaderboard submission file
        write_leaderboard(bikes_predicted);
    %end
end

function[MAE_short, baselineMAEvec, station] = profile_short()
    station = [];
    MAE_short = [];
    baselineMAEvec = [];
    %for station_num = 201:275
        a_ls = [];
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

        a_ls = [intercept_tot/200 bikes3hago_tot/200 fullprofilebikes_tot/200 fullprofile3hdiffbikes_tot/200 temp_tot/200 shortprofilebikes_tot/200 shortprofile3hdiffbikes_tot/200]'

        % Retrieve MAE and baseline MAE from estimate LS function
%         [MAE, baselineMAE] = leastSquares(a_ls, dataTable);
%         
%         station = [station; station_num];
%         
%         baselineMAEvec = [baselineMAE; baselineMAEvec]
%         MAE_short = [MAE_short; MAE];       
%         
        % load test data in a similar fashion to the training data
        dataTest = load_test_data();
        interests = {'bikes_3h_ago', 'full_profile_bikes', 'full_profile_3h_diff_bikes', 'temperatureC', 'short_profile_bikes', 'short_profile_3h_diff_bikes'};
        bikes_predicted = predict_bikes(a_ls, dataTest, interests)
        % write to the leaderboard submission file
        write_leaderboard(bikes_predicted);
    %end
end

function[MAE_temp, baselineMAEvec, station] = profile_temp()
    station = [];
    MAE_temp = [];
    baselineMAEvec = [];
    %for station_num = 201:275
        a_ls = [];
        intercept_tot = 0;
        bikes3hago_tot = 0;
        fullprofilebikes_tot = 0;
        fullprofile3hdiffbikes_tot = 0;
        temp_tot = 0;
        shortprofilebikes_tot = 0;
        shortprofile3hdiffbikes_tot = 0;

        for model_num = 1:200
            %fprintf('MAE_temp | station: %d, model: %d\n', station_num, model_num);

            % 3 profiles
            dataModelFullTemp = load_model_full_temp(model_num);
            dataModelShortFullTemp = load_model_short_full_temp(model_num);
            dataModelShortTemp = load_model_short_temp(model_num);

            %dataTable = load_training_data(station_num);
            
            % construct a_ls coefficients
            interceptMean = (dataModelFullTemp.weight(1) + ...
                dataModelShortFullTemp.weight(1) + ...
                dataModelShortTemp.weight(1))/3;
                        
            bikes3hagoMean = (dataModelFullTemp.weight(2) + ...
                dataModelShortFullTemp.weight(2) + ...
                dataModelShortTemp.weight(2))/3;
                        
            shortprofilebikesMean = (dataModelShortFullTemp.weight(3) + ...
                dataModelShortTemp.weight(3))/2;
            
            shortprofile3hdiffbikesMean = (dataModelShortFullTemp.weight(4) + ...
                dataModelShortTemp.weight(4))/2;
            
            fullprofilebikesMean = (dataModelFullTemp.weight(3) + ...
                dataModelShortFullTemp.weight(5))/2;
            
            fullprofile3hdiffbikesMean = (dataModelFullTemp.weight(4) + ...
                dataModelShortFullTemp.weight(6))/2;
            
            tempMean = (dataModelFullTemp.weight(5) + ...
                dataModelShortFullTemp.weight(7) + ...
                dataModelShortTemp.weight(5))/3;
            
            intercept_tot = intercept_tot + interceptMean;
            bikes3hago_tot = bikes3hago_tot + bikes3hagoMean;
            fullprofilebikes_tot = fullprofilebikes_tot + fullprofilebikesMean;
            fullprofile3hdiffbikes_tot = fullprofile3hdiffbikes_tot + fullprofile3hdiffbikesMean;
            temp_tot = temp_tot + tempMean;
            shortprofilebikes_tot = shortprofilebikes_tot + shortprofilebikesMean;
            shortprofile3hdiffbikes_tot = shortprofile3hdiffbikes_tot + shortprofile3hdiffbikesMean;
        end

        a_ls = [intercept_tot/200 bikes3hago_tot/200 fullprofilebikes_tot/200 fullprofile3hdiffbikes_tot/200 temp_tot/200 shortprofilebikes_tot/200 shortprofile3hdiffbikes_tot/200]'

        % Retrieve MAE and baseline MAE from estimate LS function
        %[MAE, baselineMAE] = leastSquares(a_ls, dataTable);
        
        %station = [station; station_num];
        
        %baselineMAEvec = [baselineMAE; baselineMAEvec]
        %MAE_temp = [MAE_temp; MAE];       
        
        % load test data in a similar fashion to the training data
        dataTest = load_test_data();
        interests = {'bikes_3h_ago', 'full_profile_bikes', 'full_profile_3h_diff_bikes', 'temperatureC', 'short_profile_bikes', 'short_profile_3h_diff_bikes'};
        bikes_predicted = predict_bikes(a_ls, dataTest, interests)
        % bikes_predicted = round(X*a_ls)
        %bikes_predicted(bikes_predicted < 0) = 0;
        %size(bikes_predicted);
        % write to the leaderboard submission file
        write_leaderboard(bikes_predicted);
    %end
end

function[MAE, baselineMAE] = leastSquares(a_ls, dataTable)
% construct feature matrix
        X = [dataTable.bikes_3h_ago dataTable.full_profile_bikes dataTable.full_profile_3h_diff_bikes dataTable.temperatureC dataTable.short_profile_bikes dataTable.short_profile_3h_diff_bikes];
        y = dataTable.bikes;
        y_baseline = dataTable.bikes_3h_ago;

        b = isnan(y);
        for i = 1:5
            b = b | isnan(X(:,i));
        end
        X(b,:) = [];
        y(b) = [];
        y_baseline(b) = [];


        X = [ones(size(X, 1), 1) X];
        bikes = X*a_ls;
        %bikes(bikes < 0) = 0;

        baselineMAE = mean(abs(y - y_baseline));
        MAE = mean(abs(y - bikes));
end

% load models from CSV files, and preprocess
function [dataTable] = load_model_full(num) % num - number of model files 1 - 200
    % if that file has already been created, use it without reloading
    if exist(sprintf('model_station_%d_rlm_full.csv', num), 'file')
        dataTable = readtable(sprintf('model_station_%d_rlm_full.csv', num));
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTable = importmodel(sprintf('Models/model_station_%d_rlm_full.csv', num));
        writetable(dataTable, sprintf('model_station_%d_rlm_full.csv', num));
    end
end
function [dataTable] = load_model_full_temp(num) % num - number of model files 1 - 200
    % if that file has already been created, use it without reloading
    if exist(sprintf('model_station_%d_rlm_full_temp.csv', num), 'file')
        dataTable = readtable(sprintf('model_station_%d_rlm_full_temp.csv', num));
    else
        % load the file into a table
        % changes weekdays to numbers; and NA to NaN
        dataTable = importmodel(sprintf('Models/model_station_%d_rlm_full_temp.csv', num));
        writetable(dataTable, sprintf('model_station_%d_rlm_full_temp.csv', num));
    end
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