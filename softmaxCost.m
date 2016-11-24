function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2); %num of training examples

groundTruth = full(sparse(labels, 1:numCases, 1)); % groundTruth(i,j) = 1 if y(j) = i else o
cost = 0;

thetagrad = zeros(numClasses, inputSize);


M = theta*data;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
M = bsxfun(@rdivide, M, sum(M));
S = groundTruth - M;
for j = 1:numClasses
    thetagrad(j,:) = lambda*theta(j,:) - numCases.\transpose(sum(data.*repmat(S(j,:),inputSize,1),2));
end

cost = 0.5*lambda*sum(sum(theta.*theta)) - sum(sum(log(M).*groundTruth))/numCases;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

