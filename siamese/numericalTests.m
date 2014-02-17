% Snippets for numerically checking gradients.
% Obsolete as of 2014-01-05 when the annoying misspelled-variable bug was
% eliminated.

%%{
% epsilon = 1e-7;

%dLdEw = (1-t).*5.54.*exp(-2.77/Q * Ew) / trainNum; % scalar
% numerical [VERIFIED TO BE SAME AS ANALYTICAL!]
%dLdEwng = (L(Ew + epsilon, t) - L(Ew - epsilon, t))/(2*epsilon);

% numerical [VERIFIED TO BE SAME AS ANALYTICAL!]

dEwdGng = nan(trainNum, numOutputs);
%dEwdG1ng = nan(trainNum, numOutputs);
%dEwdG2ng = nan(trainNum, numOutputs);
for ind = 1:numOutputs
    epsilonVec = zeros(1, numOutputs);
    epsilonVec(ind) = epsilon;
    epsilonVec = repmat(epsilonVec, [trainNum 1]);
    %dEwdG1ng(:,ind) = (Ew_fun(g1 + epsilonVec, g2) - Ew_fun(g1 - epsilonVec, g2)) / (2*epsilon);
    %dEwdG2ng(:,ind) = (Ew_fun(g1, g2 + epsilonVec) - Ew_fun(g1, g2 - epsilonVec)) / (2*epsilon);
    dEwdG1ng = (Ew_fun(g1 + epsilonVec, g2) - Ew_fun(g1 - epsilonVec, g2)) / (2*epsilon);
    dEwdG2ng = (Ew_fun(g1, g2 + epsilonVec) - Ew_fun(g1, g2 - epsilonVec)) / (2*epsilon);
    dEwdGng(:,ind) = (dEwdG1ng + (-dEwdG2ng)) / 2;
end

% dGdW
% numerical
%%{
dG1dWng = cell(size(nn1.W));
dG2dWng = cell(size(nn2.W));
for layer=1:numInterfaces
    dG1dWng{layer} = nan([size(nn1.W{layer}) numExamples numOutputs]);
    dG2dWng{layer} = nan([size(nn2.W{layer}) numExamples numOutputs]);
end
for layer=1:numInterfaces
    for i=1:size(dG1dWng{layer}, 1)
        for j=1:size(dG1dWng{layer}, 2)
            wij = nn1.W{layer}(i,j);

            % Do nudging of wij both upwards and downwards
            dG1dWngij = nan(2, numExamples, numOutputs);
            dG2dWngij = nan(2, numExamples, numOutputs);
            for s=[-1 1]
                nn1.W{layer}(i,j) = wij + s*epsilon;
                nn2.W{layer}(i,j) = wij + s*epsilon;

                % Save the response of G1 and G2 at each nudge
                nn1ng = nnff(nn1, x1, 0);
                nn2ng = nnff(nn2, x2, 0);

                dG1dWngij((s+1)/2 + 1,:,:) = nn1ng.a{end};
                dG2dWngij((s+1)/2 + 1,:,:) = nn2ng.a{end};
            end

            dG1dWng{layer}(i,j,:,:) = (dG1dWngij(2,:,:) - dG1dWngij(1,:,:)) / (2*epsilon);
            dG2dWng{layer}(i,j,:,:) = (dG2dWngij(2,:,:) - dG2dWngij(1,:,:)) / (2*epsilon);

            nn1.W{layer}(i,j) = wij;
        end
    end
end
dGdWng = cell(size(dG1dWng));
for layer=1:length(dG1dWng)
    dGdWng{layer} = (dG1dWng{layer} + dG2dWng{layer}) / 2;
end
%}

% Compute dLdW (using some mix of derivatives) ... Doesn't work since
% dGdW isn't correctly computed
%%{
dLdW = cell(size(nn1.W));
for layer=1:numInterfaces
    ws1 = size(nn1.W{layer},1);
    ws2 = size(nn1.W{layer},2);
    dLdW{layer} = zeros([ws1 ws2 trainNum]);
    for i=1:ws1
        for j=1:ws2
            dLdW{layer}(i,j,:) = dLdW{layer}(i,j,:) + reshape(...
                sum(...
                reshape(dGdWng{layer}(i,j,:,:),[trainNum numOutputs]) .* ...
                dEwdG,2) .* ...
                dLdEw, ...
                [1 1 100]);
        end
    end
    %dLdW{layer} = dLdW{layer} / trainNum;
end
% Set dW
for layer=1:numInterfaces
    nn1.dW{layer} = mean(dLdW{layer},3);
    nn2.dW{layer} = mean(dLdW{layer},3);
end
%}

%         dL/dW
% Numerical, whole system (last resort)
%%{
dLdWng = cell(size(nn1.W));
for layer=1:numInterfaces
    dLdWng{layer} = nan([size(nn1.W{layer}) numExamples]);
end  
for layer=1:length(dLdWng)
    %fprintf('Layer %d\n',layer);
    ws1 = size(nn1.W{layer},1);
    ws2 = size(nn1.W{layer},2);
    for i=1:ws1
        fprintf('%d/%d\n',i,size(dLdWng{layer},1));
        for j=1:ws2
            wij = nn1.W{layer}(i,j);

            Le = nan(2,numExamples);
            for s=[-1 1]
                nn1.W{layer}(i,j) = wij + s*epsilon;
                nn2.W{layer}(i,j) = wij + s*epsilon;

                nn1ng = nnff(nn1, x1, 0);
                nn2ng = nnff(nn2, x2, 0);
                g1ng = nn1ng.a{end};
                g2ng = nn2ng.a{end};
                Le((s+1)/2 + 1,:) = L_fun(Ew_fun(g1ng, g2ng), t);
            end

            dLdWng{layer}(i,j,:) = (Le(2,:) - Le(1,:)) / (2*epsilon);

            nn1.W{layer}(i,j) = wij;
            nn2.W{layer}(i,j) = wij;
        end
    end
end

%}


%
%   N U M E R I C A L   T R I A L  &  E R R O R   S E C T I O N
%
% Shows, no proofs
%
% This is to show that it's correct to use repmat and .*
% when chaining derivatives together
%
% 1. dLdG = dLdEw * dEwdG
%%{
dLdG = repmat(dLdEw, [1 numOutputs]) .* dEwdG;
% SHOW HERE, SAME RESULT WHEN DOING NUMERICAL DIFFERENTIATION
%%{
dLdGng = nan(trainNum, numOutputs);
dLdG1ng = nan(trainNum, numOutputs);
dLdG2ng = nan(trainNum, numOutputs);
for ind = 1:numOutputs
    epsilonVec = zeros(1, numOutputs);
    epsilonVec(ind) = epsilon;
    epsilonVec = repmat(epsilonVec, [trainNum 1]);
    dLdG1ng(:,ind) = (L(Ew_fun(g1 + epsilonVec, g2),t) - L(Ew_fun(g1 - epsilonVec, g2),t)) / (2*epsilon);
    dLdG2ng(:,ind) = (L(Ew_fun(g1, g2 + epsilonVec),t) - L(Ew_fun(g1, g2 - epsilonVec),t)) / (2*epsilon);
    %dLdG1ng = (L(Ew_fun(g1 + epsilonVec, g2),t) - L(Ew_fun(g1 - epsilonVec, g2),t)) / (2*epsilon);
    %dLdG2ng = (L(Ew_fun(g1, g2 + epsilonVec),t) - L(Ew_fun(g1, g2 - epsilonVec),t)) / (2*epsilon);
    %dLdGng(:,ind) = (dLdG1ng + -dLdG2ng) / 2;
end

shouldBeZero = sum(sum(abs(dLdG1ng))) - sum(sum(abs(dLdG2ng))); % same values (expr = zero)
shouldBeZero = sum(sum(abs(dLdG1ng + dLdG2ng))); % same values, opposite signs (expr = zero)

dLdGng = (dLdG1ng + -dLdG2ng) / 2;
%}
%}    

%         dEw/dW
% Numerical
%%{
dEwdWng = cell(size(nn1.W));
for layer=1:numInterfaces
    dEwdWng{layer} = nan([size(nn1.W{layer}) trainNum]);
end  
for layer=1:length(dEwdWng)
    %fprintf('Layer %d\n',layer);
    ws1 = size(nn1.W{layer},1);
    ws2 = size(nn1.W{layer},2);
    for i=1:ws1
        %fprintf('%d/%d\n',i,size(dLdWng{layer},1));
        for j=1:ws2
            wij = nn1.W{layer}(i,j);

            Le = nan(2,trainNum);
            for s=[-1 1]
                nn1.W{layer}(i,j) = wij + s*epsilon;
                nn2.W{layer}(i,j) = wij + s*epsilon;

                nn1ng = nnff(nn1, x1, 0);
                nn2ng = nnff(nn2, x2, 0);
                g1ng = nn1ng.a{end};
                g2ng = nn2ng.a{end};
                Le((s+1)/2 + 1,:) = Ew_fun(g1ng, g2ng);
            end

            dEwdWng{layer}(i,j,:) = (Le(2,:) - Le(1,:)) / (2*epsilon);

            nn1.W{layer}(i,j) = wij;
            nn2.W{layer}(i,j) = wij;
        end
    end
end

% dLdW / dEwdW = (dLdEw * dEwdW) / dEwdW = dLdEw
dLdEwng2 = dLdWng{1} ./ dEwdWng{1};
% dLdEw / dLdEw = 1
shouldBeOnes = dLdEwng2 ./ repmat(reshape(dLdEw,[1 1 100]),[10 37 1]);
closeToZero = sum(abs((shouldBeOnes(:) - 1))) / numel(shouldBeOnes);
%}

% TODO:
% Try to find dGdW using the fact that dLdW / (dLdEw * dEwdG) = dGdW
% Problem: dLdW is R^numWeights -> R,
% but dGdW is R^numWeights -> R^numOutputs
% and (dLdEw * dEwdG) is R^numOutputs -> R
% 

% ... dLdG = (repmat(dLdEw,[1 numOutputs]) .* dEwdG);


% Naive way of separating output vectors. Doesn't work.
%%{
direction = repmat(sign(t - 0.5),[1 numOutputs]); % 1 = sim, -1 = dis
g2Tog1 = (g1 - g2);
g1Tog2 = -g2Tog1;
G1error =  1/2 * g1Tog2 .* direction;
G2error =  1/2 * g2Tog1 .* direction;
%}

% Final check that system's numerical gradient is equal to computed
%%{
for layer=1:numInterfaces
    dWcorrect = mean(dLdWng{layer},3);
    shouldBeZero = dGdW{layer} - dWcorrect;
    sum(abs(shouldBeZero(:)))
end
%}
%}