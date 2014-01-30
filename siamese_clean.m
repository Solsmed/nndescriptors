clear
path = '/Volumes/Solar Flare/liberty/';
%datasetPath = 'liberty/';

[ patchList, patchSize, patchesLeft, patchesRight, patchesSimilarity ] = ...
    loadData('load',-1,'m50_10000_10000_0.mat',28);
    %loadData('load',-1,'saveName');
    %loadData('liberty',10000,path);
    %loadData('generate2D',1000,path,32);
    
%% Look at 1D data example m
m = 45;

subplot(1,2,1)
plot(patchList(m,patchesLeft))
xlim([1 patchSize^2])
ylim([0 1])

subplot(1,2,2)
plot(patchList(m,patchesRight))
xlim([1 patchSize^2])
ylim([0 1])

if(patchList(m,end))
    bgcol = [0 1 0];
else
    bgcol = [1 0 0];
end
set(gcf,'Color',bgcol)
colormap(gray(256))
   
%% Look at 2D data example m
m = 706;
subplot(1,2,1)
image(255*reshape(patchList(m,patchesLeft),[patchSize patchSize]))
subplot(1,2,2)
image(255*reshape(patchList(m,patchesRight),[patchSize patchSize]))
if(patchList(m,patchesSimilarity))
    bgcol = [0 1 0];
else
    bgcol = [1 0 0];
end
set(gcf,'Color',bgcol)
colormap(gray(256))

%% Initialise training
addpath('../DeepLearnToolbox/util')
addpath('../DeepLearnToolbox/NN')
addpath('../DeepLearnToolbox/CNN')

numOutputs = 64;
numTrainingExamples = 9000;
batchSize = 10;
numBatches = floor(numTrainingExamples / batchSize);
numEpochs = 1;
numTestingExamples = min(500,size(patchList,1) - numTrainingExamples);
numIters = numEpochs * numBatches;

fprintf('\nOutputs: %d\nExamples: %d\n  batchSize: %d\n  batches: %d\n  Epochs:%d\nTests: %d\n',numOutputs,numTrainingExamples,batchSize,numBatches,numEpochs,numTestingExamples)

seed = randperm(size(patchList,1));
trainSet = seed(1:numTrainingExamples);
testSet = seed(numTrainingExamples + (1:numTestingExamples));
x1 = patchList(trainSet,patchesLeft);
x2 = patchList(trainSet,patchesRight);
t = patchList(trainSet,patchesSimilarity);

t_x1 = patchList(testSet,patchesLeft);
t_x2 = patchList(testSet,patchesRight);
t_t = patchList(testSet,patchesSimilarity);

%
%   ********************** S I A M E S E   H E A D ************************
%

%Traditional logarithmic cost function
%L = sum( (1-t).*log(1-Ew) + t.*log(Ew) );
% Cost function, LeCun's L
% Notre Damme labels are inversed compared to LeCun's formula
% LeCun: Y = 1 <=> impostor, Y = 0 <=> genuine
% ND:    t = 1 <=> similar, t = 0 <=> dissimilar
Q = numOutputs; % upper bound for Ew

% Descriptor distance, L1 norm, not normalised
Ew_fun = @(v1, v2) sum(abs((v1 - v2)),2);
% L
L_fun = @(Ew, t) ((t).*(2/Q).*(Ew.^2) + (1-t).*2*Q.*exp(-2.77/Q * Ew));
L_funvL = @(v1, v2, t) L_fun(Ew_fun(v1, v2), t);
% dL/dEw
dLdEw_fun = @(Ew_val, t_val) ((t_val).*4/Q.*Ew_val - (1-t_val).*5.54.*exp(-2.77/Q * Ew_val)); % scalar
% dEw/dG
dEwdG_fun = @(g1_val, g2_val) sign(g1_val - g2_val);

%%
%{
%   ************ V A N I L L A   N E U R A L   N E T W O R K **************
%
hist_L = nan(numIters, 1);

architecture = [patchSize^2 1024 numOutputs];

nn = nnsetup(architecture);
nn.activation_function = 'sigm';
nn.momentum = 0.5;
numInterfaces = nn.n - 1;

nn1 = nn;
nn2 = nn;

for epoch=1:numEpochs
    fprintf('%d/%d\n',epoch,numEpochs) %progress
    
    % Mix the ordering of the inputs each epoch
    trainSet = trainSet(randperm(length(trainSet)));
    
    for batch=1:numBatches
        fprintf('    %d/%d ',batch,numBatches) %progress
        iter = (epoch - 1) * numBatches + batch;
        
        batchRange = (batch - 1) * batchSize + 1:batch * batchSize;
        x1nn = x1(batchRange,:);
        x2nn = x2(batchRange,:);
        tnn = t(batchRange,:);

        %
        %   F O R W A R D   P A S S
        %

        % Feed forward NN twins.
        % Last argument is label, but we don't care about it here,
        % we're just after the responses of the networks
        nn1 = nnff(nn1, x1nn, 0);
        nn2 = nnff(nn2, x2nn, 0);
        fprintf('.') %progress

        % Each twin's ouput vector
        g1 = nn1.a{nn1.n};
        g2 = nn2.a{nn2.n};

        Ew = Ew_fun(g1, g2);
        hist_L(iter) = mean(L_fun(Ew, tnn)) / (2*numOutputs);

        %
        %   B A C K P R O P A G A T I O N   (D E R I V A T I V E S)
        %

        % dL/dEw
        dLdEw = dLdEw_fun(Ew, tnn);   
        % dEw/dG
        dEwdG = dEwdG_fun(g1, g2);    
        % dG/dW
        %
        % algorithmical
        %%{
        dGdW = cell(size(nn1.W));
        G1error =  repmat(dLdEw,[1 numOutputs]) .* dEwdG;
        G2error = -repmat(dLdEw,[1 numOutputs]) .* dEwdG;
        nn1.e = -G1error; % nnbp uses -nn.e for delta computation
        nn2.e = -G2error; % but our derivative uses the positive error
        nn1 = nnbp(nn1);
        nn2 = nnbp(nn2);
        dG1dW = nn1.dW;
        dG2dW = nn2.dW;
        for layer=1:numInterfaces
            dGdW{layer} = dG1dW{layer} + dG2dW{layer};
        end

        fprintf('.') % progress

        %
        %   V I S U A L I S A T I O N
        %
        figure(1)
        clf
        set(gcf,'Color',[0.8 0.8 0.8])

        subplot(2,2,1)
        image(256*mat2gray(nn1.W{1}))
        colormap(jet(256))
        xlabel('input neuron')
        ylabel('output neuron')

        subplot(2,2,2)
        plot(hist_L)
        xlabel('iteration')
        ylabel('Cost')
        if (numIters > 1)
            xlim([1 numIters])
        end
        ylim([0 1])

        %if (mod(batch,10) == 1)
            subplot(2,2,3)
            [threshold, truePos, trueNeg, falsePos, falseNeg] = testSiamese(nn1, @(a, b) Ew_fun(a, b)/numOutputs, t_x1, t_x2, t_t);
        %end

        subplot(2,2,4)
        plot(1*[1 1],truePos*[0 1],'g+-'), hold on
        plot(2*[1 1],trueNeg*[0 1],'gx-')
        plot(3*[1 1],falsePos*[0 1],'r+-')
        plot(4*[1 1],falseNeg*[0 1],'rx-')
        xlim([0.5 4.5])
        ylim([0 100])
        set(gca,'XTickLabel',{sprintf('TP %.0f%%',truePos), sprintf('TN %.0f%%',trueNeg), sprintf('FP %.0f%%',falsePos), sprintf('FN %.0f%%',falseNeg)});
        ylabel('Percent')
        title('Confusion')
        drawnow

        %{
        subplot(3,3,1)
        hist(nn1.W{1}(:))
        title(sprintf('Mean %.5f',mean(nn1.W{1}(:))))
        subplot(3,3,2)
        hist(nn1.W{2}(:))
        title(sprintf('Mean %.5f',mean(nn1.W{2}(:))))
        subplot(3,3,3)
        hist(nn1.W{3}(:))
        title(sprintf('Mean %.5f',mean(nn1.W{3}(:))))

        subplot(3,3,4)
        hist(nn1.dW{1}(:))
        title(sprintf('Mean %.5f',mean(nn1.dW{1}(:))))
        subplot(3,3,5)
        hist(nn1.dW{2}(:))
        title(sprintf('Mean %.5f',mean(nn1.dW{2}(:))))
        subplot(3,3,6)
        hist(nn1.dW{3}(:))
        title(sprintf('Mean %.5f',mean(nn1.dW{3}(:))))

        subplot(3,3,7)
        image(256*mat2gray(nn1.W{1}))
        subplot(3,3,8)
        image(256*mat2gray(nn1.W{2}))
        subplot(3,3,9)
        image(256*mat2gray(nn1.W{3}))

        colormap(jet(256))

        drawnow
        %}

        %
        %   G R A D I E N T   A P P L I C A T I O N
        %

        % Set dW
        % Choose which gradient to use from the code above
        % dGdW is analytical
        % dLdWng is numerical
        for layer=1:numInterfaces
            nn1.dW{layer} = dGdW{layer};
            %nn1.dW{layer} = mean(dLdWng{layer},3); % mean over all examples

            nn2.dW{layer} = nn1.dW{layer}; % same weights
        end

        nn1 = nnapplygrads(nn1);
        nn2 = nnapplygrads(nn2);

        fprintf('!')
        fprintf(' Cost %.5f',hist_L(iter) * numOutputs)
        fprintf('\n')
    end
end

%}
%%

x1cnn = permute(reshape(x1,[numTrainingExamples patchSize patchSize]),[2 3 1]);
x2cnn = permute(reshape(x2,[numTrainingExamples patchSize patchSize]),[2 3 1]);
t_x1cnn = permute(reshape(t_x1,[numTestingExamples patchSize patchSize]),[2 3 1]);
t_x2cnn = permute(reshape(t_x2,[numTestingExamples patchSize patchSize]),[2 3 1]);

hist_L = nan(numIters, 1);

%
%   ******* C O N V O L U T I O N A L   N E U R A L   N E T W O R K *******
%

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 10, 'kernelsize', 7) %convolution layer   7  64->58
    struct('type', 's', 'scale', 11) %sub sampling layer                            58->29
    %struct('type', 'c', 'outputmaps', 10, 'kernelsize', 3) %convolution layer   5  29->24
    %struct('type', 's', 'scale', 3) %subsampling layer                            24->12
    %struct('type', 'c', 'outputmaps', 80, 'kernelsize', 5) %convolution layer   5 12->8
    %struct('type', 's', 'scale', 2) %subsampling layer                             8->4  
    %struct('type', 'c', 'outputmaps', 300, 'kernelsize', 4) %convolution layer  4  4->1
    %struct('type', 'c', 'outputmaps', 200, 'kernelsize', 3) %convolution layer   5 12->8
};

cnn = cnnsetup(cnn, x1cnn(:,:,1:batchSize), nan(numOutputs, 0));

cnn1 = cnn;
cnn2 = cnn;

%cnn = cnntrain(cnn, train_x, train_y, opts);
for epoch=1:numEpochs
    fprintf('%d/%d\n',epoch,numEpochs) %progress
    
    % Mix the ordering of the inputs each epoch
    trainSet = trainSet(randperm(length(trainSet)));
    
    for batch=1:numBatches
        fprintf('    %d/%d ',batch,numBatches) %progress
        iter = (epoch - 1) * numBatches + batch;
        
        batchRange = (batch - 1) * batchSize + 1:batch * batchSize;
        x1cnnbatch = x1cnn(:,:,batchRange);
        x2cnnbatch = x2cnn(:,:,batchRange);
        tcnnbatch = t(batchRange,:);
        
        %
        %   F O R W A R D   P A S S
        %
        % Feed forward NN twins.
        % Last argument is label, but we don't care about it here,
        % we're just after the responses of the networks
        cnn1 = cnnff(cnn1, x1cnnbatch);
        cnn2 = cnnff(cnn2, x2cnnbatch);
        fprintf('.') %progress

        % Each twin's ouput vector
        g1 = cnn1.o';
        g2 = cnn2.o';

        % Siamese output
        % Descriptor distance, L1 norm, normalised
        Ew = Ew_fun(g1, g2);
        hist_L(iter) = mean(L_fun(Ew, tcnnbatch)) / (2*numOutputs);

        %
        %   B A C K P R O P A G A T I O N   (D E R I V A T I V E S)
        %
        dLdEw = dLdEw_fun(Ew, tcnnbatch);
        dEwdG = dEwdG_fun(g1, g2);

        %{
            G1error =  repmat(dLdEw,[1 numOutputs]) .* dEwdG;
            G2error = -repmat(dLdEw,[1 numOutputs]) .* dEwdG;
            %cnn1.e = -G1error;
            %cnn2.e = -G2error;
            cnn1 = cnnbp(cnn1, cnn1.o - G1error'); % cnn.e = cnn.o - arg2
            cnn2 = cnnbp(cnn2, cnn2.o - G2error'); %

            % TODO: solve this issue:
            % (cnn2.o - (cnn2.o - G2error')) + (cnn1.o - (cnn1.o - G1error')) = 0
            % leads to no net change?
            % isn't cnnbp multiplying with input? yes it is.. so why is the net 0?
            % outputs are the same, duh... why do they evolve this way?
            % outputs are NOT the same in the beginning (but close)
            fprintf('.') %progress

            % Weight balancing
            dampening = 1;
            for l = 2 : numel(cnn1.layers)
                if strcmp(cnn1.layers{l}.type, 'c')
                    for j = 1 : numel(cnn1.layers{l}.a)
                        for ii = 1 : numel(cnn1.layers{l - 1}.a)
                            cnn1.layers{l}.dk{ii}{j} = (cnn1.layers{l}.dk{ii}{j} + cnn2.layers{l}.dk{ii}{j}) / dampening;
                            cnn2.layers{l}.dk{ii}{j} = cnn1.layers{l}.dk{ii}{j};
                        end
                        cnn1.layers{l}.db{j} = (cnn1.layers{l}.db{j} + cnn1.layers{l}.db{j}) / dampening;
                        cnn2.layers{l}.db{j} = cnn1.layers{l}.db{j};
                    end
                end
            end
            cnn1.dffW = (cnn1.dffW + cnn2.dffW) / dampening;
            cnn2.dffW = cnn1.dffW;
            cnn1.dffb = (cnn1.dffb + cnn2.dffb) / dampening;
            cnn2.dffb = cnn1.dffb;

            %cnnnumgradcheck_siamese(cnn1, x1cnnbatch, x2cnnbatch, tcnnbatch, L_funvL)
        %}
        cnn1 = cnnnumgradcheck_siamese(cnn1, x1cnnbatch, x2cnnbatch, tcnnbatch, L_funvL);
        cnn2 = cnn1;
        
        %
        %   V I S U A L I S A T I O N
        %
        %%{
        if (mod(iter,1) == 0)
        figure(1)
        clf
        set(gcf,'Color',[0.8 0.8 0.8])

        layer = 0;
        for l = 2 : numel(cnn1.layers)
            if strcmp(cnn1.layers{l}.type, 'c')
                layer = layer + 1;
                kernelRenders{layer} = nan(cnn1.layers{l}.kernelsize *[numel(cnn1.layers{l}.a) numel(cnn1.layers{l - 1}.a)]);
                for j = 1 : numel(cnn1.layers{l}.a)
                    jR = (((j - 1)*cnn1.layers{l}.kernelsize):(j*cnn1.layers{l}.kernelsize - 1)) + 1;
                    for ii = 1 : numel(cnn1.layers{l - 1}.a)
                        iiR = (((ii - 1)*cnn1.layers{l}.kernelsize):(ii*cnn1.layers{l}.kernelsize - 1)) + 1;
                        kernelRenders{layer}(jR, iiR) = cnn1.layers{l}.k{ii}{j};
                    end
                end
            end
        end
        subplot(2,4,1)
        image(256*mat2gray(kernelRenders{1}))
        colormap(jet(256))
        axis equal
        subplot(2,4,2)
        image(256*mat2gray(kernelRenders{2}))
        colormap(jet(256))
        axis equal

        subplot(2,2,2)
        plot(hist_L)
        xlabel('iteration')
        ylabel('Cost')
        if (numIters > 1)
            xlim([1 numIters])
        end
        ylim([0 1])

        subplot(2,2,3)
        [threshold, truePos, trueNeg, falsePos, falseNeg] = testSiamese(cnn1, @(a, b) Ew_fun(a, b)/numOutputs, t_x1cnn, t_x2cnn, t_t);

        subplot(2,2,4)
        plot(1*[1 1],truePos*[0 1],'g+-'), hold on
        plot(2*[1 1],trueNeg*[0 1],'gx-')
        plot(3*[1 1],falsePos*[0 1],'r+-')
        plot(4*[1 1],falseNeg*[0 1],'rx-')
        xlim([0.5 4.5])
        ylim([0 100])
        set(gca,'XTickLabel',{sprintf('TP %.0f%%',truePos), sprintf('TN %.0f%%',trueNeg), sprintf('FP %.0f%%',falsePos), sprintf('FN %.0f%%',falseNeg)});
        ylabel('Percent')
        drawnow
        end
        %}

        %
        %   G R A D I E N T   A P P L I C A T I O N
        %    
        opts.alpha = 0.5;
        cnn1 = cnnapplygrads(cnn1, opts);
        cnn2 = cnnapplygrads(cnn2, opts);

        cnn1.L = hist_L(iter);
        cnn2.L = hist_L(iter);
        %{
        if isempty(cnn1.rL)
            cnn1.rL(1) = cnn1.L;
        end
        cnn1.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        cnn2.rL = cnn1.rL;
        %}
        fprintf('!')
        fprintf(' Cost %.5f',hist_L(iter))
        fprintf('\n')
    end
end