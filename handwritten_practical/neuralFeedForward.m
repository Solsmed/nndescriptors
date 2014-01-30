function [ h, nn ] = neuralFeedForward( nn, data_x )
    % Stimulate input
    nn.a{1}(2:end) = data_x';

    for layer=1:(nn.num_layers-2)
        z = nn.theta{layer} * nn.a{layer};
        nn.a{layer+1}(2:end) = nn.g(z);
    end
    z = nn.theta{end} * nn.a{end-1};
    nn.a{end} = nn.g(z);

    h = nn.a{end}';
end

