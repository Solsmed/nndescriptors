function [ cost, numGradientVector, nn ] = neuralHypothesisNumerical( thetaVector, data_x, data_y, architecture )

    %global gradHist gradHistT;

    %if (numel(gradHist) == 0)
    %    gradHist = nan(length(thetaVector),1000);
    %    gradHistT = 1;
    %end

    fprintf('numerical hypothesis... ')

    [cost, gradientVector, nn] = neuralHypothesis(thetaVector, data_x, data_y, architecture);
    
    numGradientVector = nan(size(thetaVector));
    e = 1e-6;
    for t=1:length(thetaVector)
        epsilon = zeros(size(thetaVector));
        epsilon(t) = e;
        [cost_tp, gradientVectorP, nnp] = neuralHypothesis(thetaVector+epsilon, data_x, data_y, architecture);
        [cost_tm, gradientVectorM, nnm] = neuralHypothesis(thetaVector-epsilon, data_x, data_y, architecture);
        
        numGradientVector(t) = (cost_tp - cost_tm) / (2*e);
    end
       
    %gradHist(:,gradHistT) = numGradientVector;
    %gradHistT = gradHistT + 1;
    
    fprintf('cost: %.3f\n', cost)
end

