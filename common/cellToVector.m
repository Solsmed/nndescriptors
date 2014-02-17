function [ vec ] = cellToVector( c )
    tot_numel = 0;
    
    for i=1:length(c)
        tot_numel = tot_numel + numel(c{i});
    end
    
    vec = nan(tot_numel, 1);
    tail = 0;
    for i=1:length(c)
        head = tail + 1;
        tail = head + numel(c{i}) - 1;
        vec(head:tail) = c{i}(:);
    end
end

