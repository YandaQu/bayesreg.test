location = '.\NASA_airfoil_self_noise.csv';
addpath 'C:\Users\USer\Desktop\WinterScholar_SportAnalytics\Data set\bayesreg_matlabcentral_1.8';
output_dir = 'C:\Users\USer\Desktop\WinterScholar_SportAnalytics\Data set';

load('total_data_table.mat');


total_data = total_data;

lasso_error = cell(size(total_data,1),1);
horseshoe_error = cell(size(total_data,1),1);
horseshoe_error_plus = cell(size(total_data,1),1);
lasso_error_non = cell(size(total_data,1),1);

for k = 1:size(total_data,1)
    table = total_data{k, 1};
    data = table2array(table);
    %TF = isnan(data);
    data_size = size(data);

    %error_table = cell(3,1);

    acc_10ls = zeros(10,1);
    acc_10hs = zeros(10,1);
    acc_10hs_plus = zeros(10,1);
    acc_10ls_non = zeros(10,1);

    for n = 1:10
        cv = cvpartition(data_size(1), 'KFold', 10);
        
        acc_10folds_ls = zeros(10,1);
        acc_10folds_hs = zeros(10,1);
        acc_10folds_hs_plus = zeros(10,1);
        acc_lasso = zeros(10,1);

        for j = 1:cv.NumTestSets
        % for each iteration in cross validation, get indexes of training 
        % and testing set
            trIdx = cv.training(j);
            teIdx = cv.test(j);
        
             % split traing and testing data according to indexes
            trdf_X = data(trIdx, 1:end-1);
            trdf_y = data(trIdx, end);
            tedf_X = data(teIdx, 1:end-1);
            tedf_y = data(teIdx, end);
        
            % since bayesreg cannot accept invariate data, delete these columns
            colStay = (var(trdf_X) ~= 0);
            trdf_X = trdf_X(:, colStay);
            tedf_X = tedf_X(:, colStay);
            
            test_size = size(tedf_y);
            
            % bayesian lasso
            [beta, beta0, retval] = bayesreg(trdf_X, trdf_y, 'gaussian','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'display',false);
            pred_test_y = br_predict(tedf_X, beta, beta0, retval);

            error = sum((pred_test_y.yhat - tedf_y).^2)/test_size(1);
            acc_10folds_ls(j) = error;

            % bayesian horshoe
            [beta_hs, beta0_hs, retval_hs] =  bayesreg(trdf_X, trdf_y, 'gaussian','horseshoe','nsamples',1e4,'burnin',1e4,'thin',5,'display',false);
            pred_test_y_hs = br_predict(tedf_X, beta_hs, beta0_hs, retval_hs);

            error_hs = sum((pred_test_y_hs.yhat - tedf_y).^2)/test_size(1);
            acc_10folds_hs(j) = error_hs;

            % bayesian horshoe plus
            [beta_hs_plus, beta0_hs_plus, retval_hs_plus] =  bayesreg(trdf_X, trdf_y, 'gaussian','horseshoe+','nsamples',1e4,'burnin',1e4,'thin',5,'display',false);
            pred_test_y_hs_plus = br_predict(tedf_X, beta_hs_plus, beta0_hs_plus, retval_hs_plus);

            error_hs_plus = sum((pred_test_y_hs_plus.yhat - tedf_y).^2)/test_size(1);
            acc_10folds_hs_plus(j) = error_hs_plus;

            % lassoglm
            [B,FitInfo] = lassoglm(trdf_X, trdf_y,'normal','CV',3);
            idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
            B0 = FitInfo.Intercept(idxLambdaMinDeviance);
            coef = [B0; B(:,idxLambdaMinDeviance)];
            yhat = glmval(coef, tedf_X,'identity');
            %yhatBinom = (yhat>=0.5);

            acc = sum((yhat - tedf_y).^2)/test_size(1);
            acc_lasso(j) = acc;
        end
        acc_current = mean(acc_10folds_ls);
        acc_10ls(n) = acc_current;

        acc_current_hs = mean(acc_10folds_hs);
        acc_10hs(n) = acc_current_hs;

        acc_curret_hs_plus = mean(acc_10folds_hs_plus);
        acc_10hs_plus(n) = acc_curret_hs_plus;

        acc_current_ls = mean(acc_lasso);
        acc_10ls_non = acc_current_ls;
    
    end
    lasso_error{k} = mean(acc_10ls);
    horseshoe_error{k} = mean(acc_10hs);
    horseshoe_error_plus{k} = mean(acc_10hs_plus);
    lasso_error_non{k} = mean(acc_10ls_non);    
    %disp(mean(acc_10lasso))
end

save(sprintf('%s/CV_error.mat',output_dir),'lasso_error','horseshoe_error','horseshoe_error_plus','lasso_error_non');