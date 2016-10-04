function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1) X];

hidden_value = sigmoid(Theta1*X');
hidden_value = [ones(1, m); hidden_value];
predictions = sigmoid(Theta2*hidden_value);
costSum = 0;

for i = 1:m
	z_2 = Theta1*X(i, :)';				% 25*1
	a_2 = sigmoid(z_2);					% 25*1
	a_2 = [1; a_2];						% 26*1
	z_3 = Theta2*a_2;					% 10*1
	a_3 = sigmoid(z_3);					% 10*1
	yVec = zeros(num_labels, 1);		% Convert y to vector
	yVec(y(i)) = 1;						% 10*1
	delta_3 = a_3 - yVec;				% 10*1
	delta_2 = Theta2(:, 2:end)'*delta_3.*sigmoidGradient(z_2);		% 25*1
	Theta2_grad = Theta2_grad + delta_3*a_2';						% 10*26
	Theta1_grad = Theta1_grad + delta_2*X(i, :);					% 25*401
	costSum = costSum + yVec'*log(predictions(:,i)) + (1-yVec)'*log(1-predictions(:,i));
end

Theta1(:,1) = 0;		% Delete the bias terms
Theta2(:,1) = 0;
Theta1_grad = Theta1_grad/m + lambda*Theta1/m;
Theta2_grad = Theta2_grad/m + lambda*Theta2/m;
nn_params = [Theta1(:) ; Theta2(:)];
regTerm = nn_params' * nn_params;
J = -1/m*costSum + regTerm*lambda/(2*m);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
