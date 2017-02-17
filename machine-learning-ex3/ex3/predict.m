function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% [pp, p] = max(Theta2 * [ones(m, 1), Theta1 * [ones(m, 1), X]']);
% p = p'; 
% layer2
X = [ones(m, 1), X];
X = X * Theta1';
X = sigmoid(X);
% identical operations with respect to Theta1 and Theta2
X = [ones(m, 1), X];
X = X * Theta2';
X = sigmoid(X);

[pp, p] = max(X');
% p(p == 10) = 0;
% decoding, remark that we encode picture i to integer i for i=1:9, but zero to 10 
p = p';
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
