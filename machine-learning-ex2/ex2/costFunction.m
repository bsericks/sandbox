function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
tempGrad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
total=0;
for i=1:m
  %hsigma(x) = g(sigmaTx)
  %g(z) = 1/(1+e^-z)
  %hsigma(x) = 1/(1+e^(-sigmaTx))
  %fprintf("%f, %f\n", X(i,2), y(i));
  %fprintf('iteration = %f, total = %f \n', ((theta(1)+(theta(2)*X(i))-y(i))**2), total);
  hypothesis = 1/(1+e^(theta'*X(i,:)'));
  
  total=total+(y(i)*log(hypothesis) + (1-y(i))*log(1-hypothesis));
  
  for j=1:(size(theta)(1))
    tempGrad(j) = tempGrad(j)+((hypothesis)-y(i))*X(i,j);
  endfor
  
  
  %pause;
endfor

grad = tempGrad/m;


J = -(1/(m))*total;



% =============================================================

end
