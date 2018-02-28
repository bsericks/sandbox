function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples


% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
fprintf("%f %f\n", theta(1), theta(2));
J = (1/(2*m));
total=0;

totaltest=(1/(2*m))*sum((theta(1)+(theta(2)*X(:,2))-y(:)).**2);
fprintf("\n\n%f\n\n", totaltest);
for i=1:m
  %fprintf("%f, %f\n", X(i,2), y(i));
  %fprintf('iteration = %f, total = %f \n', ((theta(1)+(theta(2)*X(i))-y(i))**2), total);
  total=total+((theta(1)+(theta(2)*X(i,2))-y(i))**2);
  %pause;
endfor

J = (1/(2*m))*total;


% =========================================================================

end
