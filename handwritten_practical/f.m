function [ val, dfdx ] = f( x )
    fprintf('Call f with x = [');
    for i=1:length(x)
        fprintf('%.8f ',x(i))
    end
    fprintf(']\n');
    
    val = norm(0.1*x.^2 + sin(x));
    dfdx = 0.2*x + cos(x);
end

