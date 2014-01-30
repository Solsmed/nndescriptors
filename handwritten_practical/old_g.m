function [ response ] = g( z )
% Sigmoid function
    response = 1 ./ (1 + exp(-z));
end

