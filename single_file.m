% #########################################################################
% parameters which users need to specify

% location of the csv file, default ''
loc = 'C:/Users/yanda/Desktop/bayesreg.test/Classification/Adult Income/adult_processed.csv';
% the number of iterations to repeat, default 10
iter = 2;
% the number of folds to split for cross-validation, default 10
num_folds = 2;
% whether or not performing truncated power spline, default false
nonlinear = false;
% if performing truncated power spline, how many knots to use, default 4
num_knots = 4;
% specify the model to be used in bayesreg, default gaussian
model = 'logistic';
% specify the prior to be used in bayesreg, default lasso
prior = 'lasso';

% #########################################################################
% the main part of the program

% read the file into matlab as a numerical matrix
df = csvread(loc,1);

% if needs to perform truncated power spline
if nonlinear
    df = to_nonlinear(df);
end

% repeat for iter number of iterations
for i = 1:iter
    
    % generate cross validation partitions
    cv = cvpartition(length(df), 'KFold', num_folds);
    for j = 1:cv.NumTestSets
        % for each iteration in cross validation, get indexes of training 
        % and testing set
        trIdx = cv.training(j);
        teIdx = cv.test(j);
        
        % split traing and testing data according to indexes
        trdf_X = df(trIdx, 1:end-1);
        trdf_y = df(trIdx, end);
        tedf_X = df(teIdx, 1:end-1);
        tedf_y = df(teIdx, end);
        
        % since bayesreg cannot accept invariate data, delete these columns
        colStay = (var(trdf_X) ~= 0);
        trdf_X = trdf_X(:, colStay);
        tedf_X = tedf_X(:, colStay);
        
        % run bayesreg and show result stats
        [beta, beta0, retval] = bayesreg(trdf_X, trdf_y, model, prior, 'display', false);
        [pred, predstats] = br_predict(tedf_X, beta, beta0, retval, 'ytest', tedf_y, 'display', true);
    end
end