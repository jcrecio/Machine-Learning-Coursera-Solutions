function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

result = zeros(size(z, 1), size(z, 2))
for itemX = 1:size(z, 1)
    for itemY = 1:size(z, 2)
        result(itemX, itemY) =  1 / (1 + e^(-z(itemX, itemY)));
    end
end

g = result;

% =============================================================

end
