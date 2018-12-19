function [auc] = cal_auc(test_y, prob_y)
[~,~,~,auc] = perfcurve(test_y, prob_y, 1);