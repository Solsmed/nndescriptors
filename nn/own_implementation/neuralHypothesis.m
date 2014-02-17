function [cost, gradientVector, nn] = neuralHypothesis( thetaVector, data_x, data_y, architecture )
    
    %fprintf('analytical hypothesis... ')

    thetaCell = vectorToCell(thetaVector,architecture);
    nn = NeuralNetwork(thetaCell);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %                  Training iteration
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    num_examples = size(data_x,1);
    
    % remember output of each training example, the output vectors are rows
    h = nan(num_examples,nn.num_outputs);
    
    % Store gradient caused by each training example
    gradDelta = cell(size(nn.theta));
    for interface=1:length(nn.theta)
         gradDelta{interface} = zeros(size(nn.theta{interface}));
    end
    
    % Go through all training examples
    for i=1:num_examples        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Forward pass %%%%%%%%%
        
        %[h(i,:), nn] = neuralFeedForward(nn, data_x(i,:));
        
        % Stimulate input
        nn.a{1} = [1; data_x(i,:)'];
        % Hidden units
        for layer=1:(nn.num_layers-2)
            nn.z{layer+1} = [Inf; nn.theta{layer} * nn.a{layer}];
            nn.a{layer+1} = nn.g(nn.z{layer+1});
            %nn.z{layer+1}(1) = 1;
            %nn.a{layer+1}(1) = 1;
        end
        % Output layer
        nn.z{end} = nn.theta{end} * nn.a{end-1};
        nn.a{end} = nn.g(nn.z{end});

        % Hypothesis
        h(i,:) = nn.a{end}';
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculate error and delta gradient %%
        
        % Backward propagation
        % iterate backwards from output layer to first hidden layer
        
        % Delta
        
        nn.error = nn.a{end} - data_y(i,:)';
        nn.d{end} = nn.error .* nn.gprime(nn.z{end});
        
        % layer 2 is first non-input layer, num_layers-1 is last that we
        % haven't already done
        
        % sumproduct of outgoing synapses and thier resp. target deltas
        sumprod_residuals = (nn.theta{end}' * nn.d{end});
        % g'(a) * sum_j=1,number_of_destination_units( w_to_destination * r_destination )
        nn.d{end-1} = nn.gprime(nn.z{end-1}) .* sumprod_residuals;
        for layer=fliplr(2:(nn.num_layers-2))
            % sumproduct of outgoing synapses and thier resp. target deltas
            sumprod_residuals = (nn.theta{layer}' * nn.d{layer+1}(2:end));
            % g'(a) * sum_j=1,number_of_destination_units( w_to_destination * r_destination )
            nn.d{layer} = nn.gprime(nn.z{layer}) .* sumprod_residuals;
        end
        
        
        % Andrew Ng
        % Gradient of delta
        for layer=1:(nn.num_layers-2)
            dest_deltas = nn.d{layer+1}(2:end); % bias unit is not connected backwards
            source_activations = nn.a{layer};
            gradDelta{layer} = gradDelta{layer} + dest_deltas*source_activations';
        end
        dest_deltas = nn.d{end}; % no bias in output layer
        source_activations = nn.a{end-1};
        gradDelta{end} = gradDelta{end} + dest_deltas*source_activations';
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %                  Post-training assessment
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % See what the cost of current hypothesis is
    cost = J(nn.theta, h, data_y);
    
    % Andrew Ng
    % Calculate the gradient of the cost wrt theta
    gradientCell = cell(size(nn.theta));
    
    lambda = 0;
    for interface=1:(nn.num_layers - 1)
        th = nn.theta{interface};
        th(:,1) = 0; % Don't include thetas coming from bias units
        gradientCell{interface} = gradDelta{interface} + lambda * th;
    end
    
    gradientVector = cellToVector(gradientCell)/num_examples;
    
    fprintf('cost: %.3f\n', cost)
end

