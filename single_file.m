% #########################################################################
% parameters which users need to specify

% location of the csv file, default ''
loc = 'C:/Users/yanda/Desktop/bayesreg.test/Classification/Audit data/trial_processed.csv';
% the number of iterations to repeat, default 10
iter = 10;
% the number of folds to split for cross-validation, default 10
num_folds = 5;
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

% declare bayesian auc and negative log likelihood
bayes_auc = zeros(iter*num_folds, 1);
bayes_neglike = zeros(iter*num_folds, 1);

% declare normal logistic regression auc and negative log likelihood
normal_auc = zeros(iter*num_folds, 1);
normal_neglike = zeros(iter*num_folds, 1);

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
        [beta, beta0, retval] = bayesreg(trdf_X, trdf_y, model, prior, 'nsamples', 1e4, 'display', false);
        [pred, predstats] = br_predict(tedf_X, beta, beta0, retval, 'ytest', tedf_y, 'display', false);
        
        % append auc and negative log likelihood into bayesian lists
        bayes_auc((i-1)*num_folds + j) = predstats.auc;
        bayes_neglike((i-1)*num_folds + j) = predstats.neglike;
        %disp('The AUC of the bayesian logistic regression is:')
        %display(predstats.auc)
        %disp('The negative log likelihood of the bayesian logistic regression is:')
        %display(predstats.neglike)
        
        % run normal lasso on logistic regression
        [B,FitInfo] = lassoglm(trdf_X,trdf_y, 'binomial', 'CV', 3);
        idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
        B0 = FitInfo.Intercept(idxLambdaMinDeviance);
        coef = [B0; B(:,idxLambdaMinDeviance)];
        
        yhat = glmval(coef, tedf_X, 'logit');
        yhatBinom = (yhat >= 0.5);
        
        pred_auc = cal_auc(tedf_y, yhat);
        pred_neglike = cal_neglike(tedf_y, yhat);
        %disp('The AUC for the normal logistic regression is:')
        %display(pred_auc)
        %disp('The negative log likelihood for the normal logistic regression is:')
        %display(pred_neglike)
        
        %disp('###########################################################')
        % append auc and negative log likelihood into normal lists
        normal_neglike((i-1)*num_folds + j) = pred_neglike;
        normal_auc((i-1)*num_folds + j) = pred_auc;
    end
end

% show the average bayesian auc and negative loglikelihood
%disp('###################################################################')
%disp('###################################################################')
disp('The summarized stats for bayesian logistic regression and normal logistic regression')

disp([mean(bayes_auc), mean(bayes_neglike)])
disp([mean(normal_auc), mean(normal_neglike)])