function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%

 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));


M = theta*data;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
M = bsxfun(@rdivide, M, sum(M));
[~,pred] = max(M);
pred = transpose(pred);





% ---------------------------------------------------------------------

end

