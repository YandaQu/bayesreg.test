location = '..\data\Audit data\audit_risk.csv';
data = csvread(location, 1);
data_size = size(data);

acc_10repeats = zeros(1,1);
for n = 1:1
    cv = cvpartition(data_size(1), 'KFold', 10);
    acc_10folds = zeros(10,1);
    for i = 1:10 
        data_train_X = data(cv.training(i),1:(data_size(2)-1));
        col_left = (var(data_train_X) ~= 0);
        data_train_X = data_train_X(:,col_left);
        data_train_y = data(cv.training(i),data_size(2));
        data_test_X = data(cv.test(i),1:(data_size(2)-1));
        data_test_X = data_test_X(:,col_left);
        data_test_y = data(cv.test(i),data_size(2));

        [beta, beta0, retval] = bayesreg(data_train_X, data_train_y, 'logistic', 'horseshoe', 'display',false);
        pred_test_y = br_predict(data_test_X, beta, beta0, retval);
        
        test_size = size(data_test_y);
        acc = sum((pred_test_y.yhat == data_test_y))/test_size(1);
        acc_10folds(i) = acc;
    end
    acc_current = mean(acc_10folds);
    acc_10repeats(n) = acc_current;
end
disp(mean(acc_10repeats))