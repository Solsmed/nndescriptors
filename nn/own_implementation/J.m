function [ jVal ] = J( theta, h, y)
    
    %%{
    % Andrew Ng, Coursera Machine Learning
    num_examples = size(y,1);
    
    lambda = 0;
    
    % Error-cost term
    cost_yes = y.*log(     h);
    cost_no = (1 - y).*log(1 - h);
    % If the hypothesis is exactly 1 or exactly 0, log(h(x)) is NaN
    % which will propagate and make the cost NaN, when in fact it's
    % a good thing, apart from the fact that sigmoid can't be 1 nor 0
    % but hey, miracles happen. (and other g(z):s might be in [0..1])
    cost_yes(isnan(cost_yes)) = 0;
    cost_no(isnan(cost_no)) = 0;
    cost_inner_sum = cost_yes + cost_no;
    
    cost_inner_sum = sum(sum(cost_inner_sum));
    
    % Regularisation term
    theta_sqr_sums = 0;
    for interface=1:length(theta)
        theta_sqr_sums = theta_sqr_sums + ...
            sum(sum(theta{interface}(:,2:end).^2));
    end
    
    % Total cost
    jVal = -1/num_examples * cost_inner_sum ...
           +lambda/(2*num_examples) * theta_sqr_sums;     
    %}
    
    jVal = 1/num_examples * (1/2) * sum(sum((h - y).^2));
    
    %{
    % EPFL booklet notes, t {-1,1}
    t = (y*2-1);
    % not sure if it should be -t.*h or -t.*somethingelse
    jVal = sum(log(1 + exp(-t.*h)));
    %}
end

