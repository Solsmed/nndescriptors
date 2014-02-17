%
% Compares results of own implementation with toolbox implementation
%

addpath('common')
addpath('nn/own_implementation')
addpath('DeepLearnToolbox/NN')

clc
clear

ex_x = [1 1 1 2 1 3;
        1 0 1 1 0 1
        ];
ex_y = [1 -1 1;
        0  1 1
        ];
architecture = [length(ex_x) 5 length(ex_y)];
nnMy = NeuralNetwork(architecture);
nnTB = nnsetup(architecture);
nnTB.learningRate = 1;
nnTB.activation_function = 'sigm';
nnMy.theta = nnTB.W;

fprintf('-- Initialisation --\n')
% Initialise states equally
for layer=1:length(architecture)
    if(layer < length(architecture))
        Wdiff = (nnTB.W{layer} - nnMy.theta{layer});
        assert(norm(Wdiff(:)) < 1e-16,sprintf('Differnet weights from layer %d\n',layer))
    end
end
fprintf('OK. Networks initialised equivalently.\n')

%%
fprintf('-- Forward propagation --\n')
% Forward prop
nnTBff = nnff(nnTB, ex_x, ex_y);
[~, nnMyGrad, nnMyff] = neuralHypothesis(cellToVector(nnMy.theta), ex_x, ex_y, architecture);

% Compare states
for layer=1:length(architecture)
    if(layer < length(architecture))
        Wdiff = (nnTBff.W{layer} - nnMyff.theta{layer});
        assert(norm(Wdiff(:)) < 1e-16,sprintf('Differnet weights from layer %d\n',layer))
    end
    assert(norm(nnMyff.a{layer} - nnTBff.a{layer}(end,:)') == 0,sprintf('Different activations in layer %d\n',layer))
end
assert(norm((-nnMyff.error) - nnTBff.e(end,:)') == 0,sprintf('Different output errors\n'))
fprintf('OK. Network states equivalent after forward pass.\n')

%%
fprintf('-- Backward propagation --\n')
% Backward prop (calculate dW, though deltas and errors)
nnMybp = nnMyff;
nnTBbp = nnbp(nnTBff);

% Compare gradients
gradCell = vectorToCell(nnMyGrad, architecture);
for layer=1:length(architecture)
    if(layer < length(architecture))
        dWdiff = (nnTBbp.dW{layer} - gradCell{layer});
        assert(norm(dWdiff(:)) < 1e-16,sprintf('Differnet gradients from layer %d\n',layer))
    end
end
fprintf('OK. Network gradients equivalent after back propagation.\n')

%%
fprintf('-- Gradient application --\n')
% Apply grads
nnTB2 = nnapplygrads(nnTBbp);
nnMy2 = nnMybp;

for layer=1:(nnMybp.num_layers-1)
    nnMy2.theta{layer} = nnMybp.theta{layer} - gradCell{layer};
end

% Compare states
for layer=1:length(architecture)
    if(layer < length(architecture))
        Wdiff = (nnTB2.W{layer} - nnMy2.theta{layer});
        assert(norm(Wdiff(:)) == 0,sprintf('Differnet weights from layer %d\n',layer))
    end
    assert(norm(nnMy2.a{layer} - nnTB2.a{layer}(end,:)') == 0,sprintf('Different activations in layer %d\n',layer))
end
assert(norm((-nnMy2.error) - nnTB2.e(end,:)') == 0,sprintf('Different output errors\n'))
fprintf('OK. Network states equivalent after gradient application pass.\n')

%% numerical hypothesis

[~, nnMyGradNum, nnMyffNum] = neuralHypothesisNumerical(cellToVector(nnMy.theta), ex_x, ex_y, architecture);
assert(norm(nnMyGradNum - nnMyGrad) < 1e-4,'My numerical gradient differs from my analytical one.')
%nnMyGrad ./ nnMyGradNum - 1
fprintf('OK. Network numerical gradient consistent.\n')

