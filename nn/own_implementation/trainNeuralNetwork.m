function [ nn ] = trainNeuralNetwork(data_x, data_y, arch_nn)
    
    if(isnumeric(arch_nn))
        % Create a new network using architecture
        architecture = arch_nn;
        initial_nn = NeuralNetwork(architecture);
        initialThetaVector = cellToVector(initial_nn.theta);
    else
        % Argument is already a network, resume training
        initialThetaVector = cellToVector(arch_nn.theta);
        architecture = arch_nn.neuronCounts;
    end
    
    %options = optimoptions('GradObj','on'); % indicate gradient is provided 
    f = @(thetaVec) neuralHypothesis(thetaVec,data_x,data_y,architecture);
    %optThetaVec = fminunc(f, initialThetaVector, optimset('GradObj','on'));
    optThetaVec = gradientDescent(f, initialThetaVector, 0.001, 3000);
    
    nn = NeuralNetwork(vectorToCell(optThetaVec,architecture));
end

