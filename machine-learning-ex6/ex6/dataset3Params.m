function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%initialize the C and sigma
C_value_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_value_set = C_value_set;
error_set = zeros(1, 64); % column vector
k = 1;
for i = 1:8
	for j = 1:8
		models = svmTrain(X, y, C_value_set(i),@(x1, x2) gaussianKernel(x1, x2, sigma_value_set(j)));
		predictions = svmPredict(models, Xval);
		error_set(k) = mean(double(predictions ~= yval));
		k = k + 1;
	end
end
[max_value, index] = min(error_set);
C = C_value_set(floor(index / 8));
sigma = sigma_value_set(mod(index, 8));

% hhhh,i do not konw why i initialize those matrices soo many either

% evaluate this models







% =========================================================================

end
