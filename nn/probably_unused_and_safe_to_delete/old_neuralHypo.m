function [ h, gradDelta ] = neuralHypo( thetaVector, data_x, data_y, hidden )

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %% Build neural network and initialise everything
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %{
    num_inputs = size(data_x,2);
    num_outputs = size(data_y,2);
    num_examples = size(data_x,1);
    neuronCounts = [num_inputs hidden num_outputs]; % excluding bias neurons
    num_layers = length(neuronCounts);
    
    % Each element in neurons is the post-synaptic response of the
    % corresponding neuron in the network.
    activations = cell(num_layers,1);
    % Between each layer there is a matrix of theta values, coupling
    % each neuron of layer j to each neuron of layer j+1
    % second dimension is history
    theta = cell(num_layers-1,1);
    tail = 0;
    % deltas{1} (input layer) never used but keeping it makes nice indices
    % For back propagation
    delta = cell(num_layers,1);
    % Partial derivatives, one for each theta value
    gradDelta = cell(num_layers-1,1);
    
    
    for layer=1:(num_layers-1)
        n_from = neuronCounts(layer) + 1; % no of transmitting neurons in current layer
        n_to = neuronCounts(layer+1); % number of receptive neurons in next layer
        activations{layer} = zeros(n_from, 1);
        head = tail + 1;
        tail = head + n_to*(n_from) - 1;
        theta{layer} = reshape(thetaVector(head:tail), n_to, n_from);
        % deltas{:}(1,:) never used but keeping it makes nice indices
        delta{layer} = zeros(n_from-1, 1);
        gradDelta{layer} = zeros(n_to, n_from);
    end
    activations{end} = zeros(neuronCounts(end), 1);
    delta{end} = zeros(neuronCounts(end),1);
    
    % Turn on biases
    for layer=1:(num_layers-1)
        activations{layer}(1) = 1;
    end
    %}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %% Training iteration
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % remember output of each training example, the output vectors are rows
    h = nan(num_examples,num_outputs);
    
    % Go through all training examples
    for i=1:num_examples        
        % Stimulate input
        activations{1}(2:end) = data_x(i,:)';

        % Forward pass
        for layer=1:(num_layers-2)
            z = theta{layer} * activations{layer};
            activations{layer+1}(2:end) = g(z);
        end
        activations{num_layers} = g(theta{num_layers-1} * activations{num_layers-1});
        
        h(i,:) = activations{end}';

        % Backward propagationtheta
        % iterate backwards from output layer to first hidden layer
        % (we don't do delta on input)
        delta{num_layers} = activations{num_layers} - data_y(i,:)';
        
        % layer 2 is first non-input layer, num_layers-1 is last that we
        % haven't already done
        for layer=fliplr(2:(num_layers-1))
            gprime = (activations{layer}(2:end) .* (1 - activations{layer}(2:end)));
            % sumproduct of outgoing synapses and thier resp. target deltas
            sumprod_residuals = (theta{layer}(:,2:end)' * delta{layer+1});
            % g'(a) * sum_j=1,number_of_destination_units( w_to_destination * r_destination )
            delta{layer} = gprime .* sumprod_residuals;
        end
        
        %{
        delta{num_layers-1} = theta{num_layers-1}' * delta{num_layers} .* (activations{num_layers-1}.*(1-activations{num_layers-1}));
        for layer=fliplr(2:(num_layers-2))
            delta{layer} = theta{layer}' * delta{layer+1}(2:end) .* (activations{layer}.*(1-activations{layer}));
        end
        %}
        
        % Update gradDelta after each training example
        for layer=1:(num_layers-2)
            gradDelta{layer}(:,2:end) = gradDelta{layer}(:,2:end) + delta{layer+1}*activations{layer}(2:end)';
        end
        gradDelta{num_layers-1}(:,2:end) = gradDelta{num_layers-1}(:,2:end) + delta{num_layers}*activations{num_layers-1}(2:end)';
    end

end

