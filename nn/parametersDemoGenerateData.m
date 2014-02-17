addpath('DeepLearnToolbox/util')
addpath('DeepLearnToolbox/NN')
addpath('common')

%
% Generates data to be displayed by parametersDemoPlot1L (one-layer) or
% parametersDemoPlot2L (two-layers). Any new run will save data as
% nn_new.mat. Existing results are found in the results folder.
%

clc
clear

load('DeepLearnToolbox/data/mnist_uint8.mat')
x = double(train_x);
y = double(train_y);
tx = double(test_x);
ty = double(test_y);

%
% nc is the architectures to be tested
%
nc = {[1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 70 80 90 100]};
nc = {[10 20 30 40 50 60 70 80 90 100];
      [10 20 30 40 50 60]};
  
numArchs = 1;
for c=1:length(nc)
    numArchs = numArchs * numel(nc{c});
end

hiddenArchs = cell(numArchs,1);
arcs = allcomb(nc{:});
for h=1:numArchs
    hiddenArchs{h} = arcs(h,:);
end

numArchs = length(hiddenArchs);
numStats = 10;
numIters = 100;
numTests = size(ty,1);
numOutputs = size(y,2);
numInputs = size(x,2);
%hist_py = cell(numStats, numArchs);
hist_A = nan(numOutputs^2, numIters, numStats, numArchs);
hist_iterTime = nan(numIters, numStats, numArchs);
%hist_nn = cell(numStats, numArchs);

%%
archTimes = nan(numArchs, 1);
tic
for archIter=1:numArchs
    fprintf('Hidden architecture: ')
    fprintf('%d ',hiddenArchs{archIter});
    fprintf('\n')
    archStart = toc;
    
    for statIter=1:numStats
        statStart = toc;
        architecture = [numInputs hiddenArchs{archIter} numOutputs];
        nn = nnsetup(architecture);
        nn.learningRate = 1;
        nn.activation_function = 'sigm';

        gradNorm = Inf;
        gradTol = 1e-2;

        numParameters = 0;
        for layer=1:nn.n-1
            numParameters = numParameters + numel(nn.W{layer});
        end
        recLength = 5000;
        hist_grad = nan([recLength numParameters]);
        hist_L = nan([recLength 1]);

        iter = 0;
        while (iter < numIters) % gradNorm > gradTol && 
            iterStart = toc;
            iter = iter + 1;

            nn = nnff(nn, x, y);
            nn = nnbp(nn);
            nn = nnapplygrads(nn);

            gradVector = cellToVector(nn.dW);
            gradNorm = norm(gradVector);
            fprintf('%d.%d.%d: L=%.5f GRAD=%.5f\n',archIter,statIter,iter,nn.L,gradNorm)

            hist_grad(iter,:) = gradVector;
            hist_L(iter) = nn.L;

            %{
            clf
            subplot(2,1,1)
            WIDTH = 6;
            HEIGHT = 5;
            wGrid1 = nan(28*[WIDTH HEIGHT]);
            vecW = cellToVector(nn.W);
            wm = reshape(gradVector(1:23550),30,785);
            for s=1:30
                nm = wm(s,:);
                ni = reshape(nm(2:end),28,28);

                r = mod(s-1,WIDTH);
                c = floor((s-1)/(WIDTH));
                rRange = (28*r)+1:(r+1)*28;
                cRange = (28*c)+1:(c+1)*28;
                wGrid1(rRange,cRange) = mat2gray(ni);
            end

            WIDTH = 10;
            HEIGHT = 1;
            wGrid2 = nan([10 1].*[6 5]);
            vecW = cellToVector(nn.W);
            wm = reshape(gradVector(23551:23860),10,31);
            for s=1:10
                nm = wm(s,:);
                ni = reshape(nm(2:end),6,5);

                r = mod(s-1,WIDTH);
                c = floor((s-1)/WIDTH);
                rRange = (6*r)+1:(r+1)*6;
                cRange = (5*c)+1:(c+1)*5;
                wGrid2(rRange,cRange) = mat2gray(ni);
            end

            subplot(2,1,1)
            wGrid2r = imresize(wGrid2,2.8,'nearest');
            image(256*mat2gray([wGrid1 wGrid2r]'))
            axis equal, axis off

            subplot(2,1,2)
            %image(256*mat2gray(wGrid2'))
            %axis equal, axis off

            colormap(gray(256))
            %}
            tis = randperm(size(tx,1));
            py = nan(size(tx(tis(1:numTests),:),1),1);
            tty = nan(size(py));
            for i=1:size(py,1)
                py(i) = nnpredict(nn,tx(i,:));
                [~, tty(i)] = max(ty(i,:));
            end

            correctActual = tty(py == tty);
            %correctPrediction = py(py == tty); % same as correctActual
            incorrectActual = tty(py ~= tty); % classes hard for nn to see
            incorrectPrediction = py(py ~= tty); % network tendency
            correct = 100*length(correctActual)/length(tty);

            A = zeros(numOutputs);
            for i=1:size(py,1)
                A(tty(i),py(i)) = A(tty(i),py(i)) + 1;
            end

            %{
            subplot(2,2,1)
            image(256*mat2gray(A))%, colormap(hot)
            axis on
            title(sprintf('Prediction matrix\nAccuracy: %.1f%% Iteration: %d',correct, iter))
            ylabel('Input')
            xlabel('Prediction')

            subplot(2,2,2)
            hist(correctActual, 1:size(ty,2));
            title('Correct')

            subplot(2,2,3)
            hist(incorrectActual, 1:size(ty,2));
            title('Classes hard to correctly classify')

            subplot(2,2,4)
            hist(incorrectPrediction, 1:size(ty,2));
            title('Network tendency upon incorrect prediction')

            drawnow
            %}
            hist_A(:, iter, statIter, archIter) = A(:);
            iterTime = toc - iterStart;
            %pause(2*iterTime)
            hist_iterTime(iter, statIter, archIter) = iterTime;
        end

        %hist_nn{statIter, archIter} = nn;
        %hist_py{statIter, archIter} = py;
        statTime = toc - statStart;
        fprintf('Stat pass took %0.f seconds. Arch done in %0.f seconds.\n',statTime,statTime * (numStats - statIter));
        
        save nn_new.mat hist_A hist_iterTime nc numArchs numStats numIters numInputs numOutputs numTests
        %save -v7.3 nn_histnn.mat hist_nn
        
        fprintf('State saved. Loops are: %d.%d.%d\n',archIter, statIter, iter)
    end
    archTimes(archIter) = toc - archStart;
    fprintf('Arch took %0.f seconds. Simulation done in %0.f seconds.\n', archTimes(archIter), (numArchs-archIter)*nanmean(archTimes(1:find(~isnan(archTimes),1,'last'))));
    
    %save -v7.3 nn_new.mat hist_A hist_iterTime hist_nn nc numArchs numStats numIters numInputs numOutputs numTests
end

fprintf('Simulation done in %.0f seconds.',toc)