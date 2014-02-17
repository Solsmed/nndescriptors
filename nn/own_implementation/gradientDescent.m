function [ xMin ] = gradientDescent( fg,x0,tol,MAX_ITERS )
    xj = x0;
    g = Inf;
    iters = 0;
    while (norm(g)>tol && iters < MAX_ITERS)
        [f,g]=feval(fg,xj);
        xj = xj - g;
        iters = iters + 1;
    end
    
    if(iters >= MAX_ITERS)
        fprintf('Maximum number of iterations reached for gradient descent.\n');
    end
    xMin = xj;
end

