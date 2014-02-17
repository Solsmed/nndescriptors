function [threshold, truePos, trueNeg, falsePos, falseNeg] = testSiamese(nn, dist, t_x1, t_x2, t_t)
% Returns some metrics that describe a network's performance.
%   INPUT:
%       nn - the network to be tested
%       dist - the distance metric to be used for outputs in the joining
%              stage. A function.
%       t_x1, t_x2 - inputs for each half of the network
%       t_t - target values for each pair
%
%   OUTPUT:
%       threshold - point there likelihood for match/no match is equal.
%       truePos, trueNeg, falsePos, falseNeg - rates for TP/TN/FP/FN
%
% Function also plots this data (approximates error rates as gaussians)

numTests = size(t_t, 1);
%tests = testSet(1:numTests);

simEw = nan([numTests 1]);
disEw = nan([numTests 1]);

%patchesSimilarity = size(patchList,2);
%patchesLeft = 1:((patchesSimilarity - 1) / 2);
%patchesRight = (max(patchesLeft)+1):(patchesSimilarity - 1);

%for i = 1:numTests;
    %t_x1 = patchList(i,patchesLeft);
    %t_x2 = patchList(i,patchesRight);

    if (any(strcmp('W',fieldnames(nn))))
        nn = nnff(nn, t_x1, 0);
        t_y1 = nn.a{nn.n};

        nn = nnff(nn, t_x2, 0);
        t_y2 = nn.a{nn.n};
    elseif (any(strcmp('ffW',fieldnames(nn))))
        nn = cnnff(nn, t_x1);
        t_y1 = nn.o';
         
        nn = cnnff(nn, t_x2);
        t_y2 = nn.o';
    end
    % Descriptor distance
    %t_Ew = sqrt(sum((t_y1 - t_y2).^2,2));
    t_Ew = dist(t_y1, t_y2);

    %fprintf('Distance: %.4f%% [Similarity: %d]\n',100*t_Ew/numOutputs,patchList(i,patchesSimilarity))
    %if(t_t(i))
        simEw(t_t == 1) = t_Ew(t_t == 1);
        simEw(t_t ~= 1) = [];
    %else
        disEw(t_t == 0) = t_Ew(t_t == 0);
        disEw(t_t ~= 0) = [];
    %end
    
%end
numSims = sum(t_t);
numDis = sum(~t_t);
%fprintf('Average distacens [sim/dis]: %.4f%% / %.4f%%\n',100*simEw/numSims/numOutputs,100*disEw/numDis/numOutputs)
%fprintf('\n')

simX = linspace(min(simEw),max(simEw),100);
disX = linspace(min(disEw),max(disEw),100);
bothX = linspace(min(min(simEw),min(disEw)),max(max(simEw),max(disEw)),1000);
muSim = nanmean(simEw);
muDis = nanmean(disEw);
sigmaSim = sqrt(nanvar(simEw));
sigmaDis = sqrt(nanvar(disEw));
simY = @(sim_x) normpdf(sim_x,muSim,sigmaSim);
disY = @(dis_x) normpdf(dis_x,muDis,sigmaDis);
%na = 1 / max(simY(nanmean(simEw)),disY(nanmean(disEw)));
%na = 
%simY = @(sim_x) normpdf(sim_x,nanmean(simEw),sqrt(nanvar(simEw)));
%disY = @(dis_x) normpdf(dis_x,nanmean(disEw),sqrt(nanvar(disEw)));

plot(bothX, simY(bothX), 'g'), hold on
plot(bothX, disY(bothX), 'r')

xlim([0 max(quantile(simEw,0.95),quantile(disEw,0.95))])
plot(min(simEw),simY(min(simEw)),'g*')
plot(max(simEw),simY(max(simEw)),'g*')
plot(min(disEw),disY(min(disEw)),'r*')
plot(max(disEw),disY(max(disEw)),'r*')

ix = intersect_gaussians(muSim, muDis, sigmaSim, sigmaDis);
threshold = ix(find((ix > 0) .* (ix > muSim) .* (ix < muDis)));
if(numel(threshold) == 0)
    threshold = ix(find((ix > 0)));
    if(numel(threshold) > 1)
        threshold = threshold(end);
    end
end
plot(ix, simY(ix),'b*')

%{
truePos = sum(simEw < threshold) / sum(~isnan(simEw)) * 100;
trueNeg = sum(disEw > threshold) / sum(~isnan(simEw)) * 100;
falsePos = sum(disEw < threshold) / sum(~isnan(disEw)) * 100;
falseNeg = sum(simEw > threshold) / sum(~isnan(disEw)) * 100;
%}

truePos = sum(simEw < threshold) / numSims * 100;
trueNeg = sum(disEw > threshold) / numSims * 100;
falsePos = sum(disEw < threshold) / numDis * 100;
falseNeg = sum(simEw > threshold) / numDis * 100;

xlabel('Ew distance')
ylabel('Likelihood')
xlim([0 1])
%title(sprintf('Discriminative threshold Ew = %.3f\nTP: %.1f%%, TN: %.1f%%, FP: %.1f%%, FN: %.1f%%',threshold(threshold > 0), truePos, trueNeg, falsePos, falseNeg))

end