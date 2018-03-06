function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
tempGrad = zeros(size(theta));

n = length(theta);
total = 0;

total2 = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
  
  hypothesis = sigmoid(theta'*X(i,:)');
  
  total=total+(-y(i)*log(hypothesis) - (1-y(i))*log(1-hypothesis));
  
  for j=1:(size(theta)(1))
    tempGrad(j) = tempGrad(j)+((hypothesis)-y(i))*X(i,j);
  endfor
  
endfor

shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];
total2 = (lambda/(2*m))*(theta_reg'*theta_reg);


grad = tempGrad/m;

for j=2:(size(theta)(1))
    grad(j) = grad(j)+(lambda/m * theta(j));
endfor
    
%total2 = (lambda/(2*m))*total2;
total = total/m;
J = (total+total2);

%===========================================
h = sigmoid(X*theta);
% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];

%J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;

% grad_zero = (1/m)*X(:,1)'*(h-y);
% grad_rest = (1/m)*(shift_x'*(h - y)+lambda*shift_theta);
% grad      = cat(1, grad_zero, grad_rest);

%grad = (1/m)*(X'*(h-y)+lambda*theta_reg);


% =============================================================

end
