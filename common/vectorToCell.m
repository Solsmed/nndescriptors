function [ c ] = vectorToCell( vec, unitCount )

    num_layers = length(unitCount);
    
    c = cell(num_layers - 1,1);
    
    tail = 0;
    for layer=1:(num_layers - 1)
        n_from = unitCount(layer) + 1; % no of transmitting neurons in current layer
        n_to = unitCount(layer+1); % number of receptive neurons in next layer
        head = tail + 1;
        tail = head + n_to*n_from - 1;
        c{layer} = reshape(vec(head:tail), n_to, n_from);
    end

end

