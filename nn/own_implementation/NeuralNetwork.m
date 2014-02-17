function [ nn ] = NeuralNetwork( T )
    
    nn.num_layers = 0;
    nn.neuronCounts = [];
    nn.a = cell(0); % activations
    nn.d = cell(0); % deltas
    nn.z = cell(0); % unit-input weighted sums
    nn.theta = cell(0); % thetas
    nn.g = @(z) 1 ./ (1 + exp(-z)); % g(z) response function
    nn.gprime = @(z) nn.g(z) .* (1 - nn.g(z)); % g'(z)
    nn.error = [];
   
    if(iscell(T))
        nn.theta = T;
        nn.num_layers = length(T) + 1;
        % set neuron counts (excluding bias)
        for layer=1:(nn.num_layers - 1)
            nn.neuronCounts(layer) = size(T{layer},2) - 1;
        end
        nn.neuronCounts(nn.num_layers) = size(T{nn.num_layers - 1},1);
    end
    
    if(isnumeric(T))
        nn.neuronCounts = T;
        nn.num_layers = length(T);
        % init theta
        nn.theta = cell(1,nn.num_layers-1);
        % Initialise thetas to evenly distributed weights + noise
        for layer=1:(nn.num_layers-1)
            n_to = nn.neuronCounts(layer+1);
            n_from = nn.neuronCounts(layer) + 1; % +1 for bias
            vals = random('norm',0,0.1,n_to,n_from);
            vals(vals==0) = 0.01;
            % normalise inputs for each neuron
            %{
            lens = sum(vals.^2,2);
            vals = vals ./ repmat(lens,1,num_from);
            %}    
            nn.theta{layer} = vals;
        end
    end
    
    nn.num_inputs = nn.neuronCounts(1);
    nn.num_outputs = nn.neuronCounts(end);
    
    nn.a = cell(1, nn.num_layers);
    nn.d = cell(1, nn.num_layers);
    nn.z = cell(1, nn.num_layers);
    for layer=1:(nn.num_layers - 1)
        % include bias
        nn.a{layer} = zeros(nn.neuronCounts(layer) + 1,1);
        nn.d{layer} = zeros(nn.neuronCounts(layer) + 1,1);
        nn.z{layer} = zeros(nn.neuronCounts(layer) + 1,1);
    end
    % output layer has no bias
    nn.a{end} = zeros(nn.neuronCounts(end),1);
    nn.d{end} = zeros(nn.neuronCounts(end),1);
    nn.z{end} = zeros(nn.neuronCounts(end),1);
end

