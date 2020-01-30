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

% I was totally flummoxed by the whole a:b * c thing,
% but it is simply read as a:(b*c)
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

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
a3 = sigmoid(a2*Theta2');

yAsMatrix = eye(num_labels)(y,:);

for i = 1:m
    for k = 1:num_labels
        J += (-yAsMatrix(i,k)*log(a3(i,k))-(1-yAsMatrix(i,k))*log(1-a3(i,k)));
    end
end
J = sum(-yAsMatrix *log(a3)-(1-yAsMatrix*log(1-a3))) / m;

reg = 0;
t1s = Theta1.^2;
st1s = sum(t1s(:,2:end)(:)); 

t2s = Theta2.^2;
st2s = sum(t2s(:,2:end)(:));
J += (st1s+st2s)*lambda/(2*m);

delta3 = a3 - yAsMatrix;

delta2 = delta3*Theta2(:,2:end).*sigmoidGradient(z2);

delta2 = delta3'*a2;
D1 = delta2'*a1;

Theta2_grad += delta2/m;
Theta1_grad += D1/m;

Theta2_grad += lambda*Theta2/m;
Theta1_grad += lambda*Theta1/m;

Theta2_grad(:,1) -= lambda*Theta2(:,1)/m;
Theta1_grad(:,1) -= lambda*Theta1(:,1)/m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end