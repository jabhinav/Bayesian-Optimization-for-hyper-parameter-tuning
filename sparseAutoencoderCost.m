function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));


% 
size_d = size(data, 2);
m_a2 = zeros(hiddenSize,size_d);
m_h = zeros(visibleSize,size_d);
rho = zeros(hiddenSize,1);

% vectorized implementation
Z2 = W1*data + repmat(b1,1,size_d);
a2 = sigmoid(Z2);
Z3 = W2*a2 + repmat(b2,1,size_d);
h = sigmoid(Z3);
rho = size_d .\ sum(a2,2);
rho = repmat(rho,1,size_d);


% vectorized implementation
a3 = h;
delta3 = -(data-a3).*(a3.*(1-a3));
delta2 = ((transpose(W2)*delta3) + beta.*((-sparsityParam)./rho + (1-sparsityParam)./(1-rho))).*(a2.*(1-a2));
W1grad = delta2*transpose(data);
W2grad = delta3*transpose(a2); 
b1grad = sum(delta2,2);
b2grad = sum(delta3,2);
cost = 0.5*sum(sum((a3 - data).*(a3 - data),1));

W1grad = W1grad./size_d + lambda.*W1;
W2grad = W2grad./size_d + lambda.*W2;
b1grad = b1grad./size_d ;
b2grad = b2grad./size_d ;

% make sure to check formula for sparsity_penalty in case of vectorization
cost = cost/size_d;
weight_decay = (lambda*0.5)*( sum(sum(W1.*W1)) + sum(sum(W2.*W2)) );
sparsity_penalty = beta * sum(sparsityParam.*log(sparsityParam./rho(:,1)) + (1-sparsityParam).*log((1-sparsityParam)./(1-rho(:,1))));
cost = cost + weight_decay + sparsity_penalty;




grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end


function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

