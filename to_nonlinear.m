function [output_data] = to_nonlinear(input_data)

linear = input_data;
poly2 = input_data.^2;

quantile25 = quantile(input_data, 0.25);
quantile50 = quantile(input_data, 0.5);
quantile75 = quantile(input_data, 0.75);

data25 = input_data - quantile25;
data50 = input_data - quantile50;
data75 = input_data - quantile75;

data25(data25 < 0) = 0;
data50(data50 < 0) = 0;
data75(data75 < 0) = 0;

output_data = [linear, poly2, data25, data50, data75];