%
%   Code to generate performance data for (toolbox) CNN.
%   Performace tests relation between batch size and learning speed.
%   Outputs one mat file for each batchSize setting.
%

addpath('DeepLearnToolbox/data')
addpath('DeepLearnToolbox/CNN')
addpath('DeepLearnToolbox/util')

load mnist_uint8


train_x = permute(double(reshape(train_x',28,28,60000))/255, [2 1 3]);
test_x = permute(double(reshape(test_x',28,28,10000))/255, [2 1 3]);
train_y = double(train_y');
test_y = double(test_y');

numTrainingExamples = 60000;
%numTestingExamples = 1000;
%testPerm = randperm(size(test_y,2));
testPerm = 501:1300;
numTestingExamples = length(testPerm);

xs = train_x(:,:,1:numTrainingExamples);
ys = train_y(:,1:numTrainingExamples);
t_xs = test_x(:,:,1:numTestingExamples);
t_ys = test_y(:,1:numTestingExamples);
numOutputs = 10;
    
numEpochs = 1;

for batchSize = [2 3 5 8 12 32 48 80 125 200 300 400 600 800 1000]
    numBatches = floor(numTrainingExamples / batchSize);

    numIterations = numEpochs * numBatches;

    hist_L = nan(numIterations,1);
    %hist_Ignd = nan(numIterations,numTestingExamples);
    hist_o = nan(numIterations,numOutputs,numTestingExamples);
    hist_time = nan(numIterations,1);
    
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
    };

    cnn = cnnsetup(cnn, xs, nan(numOutputs, 0));

    for epoch = 1:numEpochs
        fprintf('%d/%d\n',epoch,numEpochs)
        for batch = 1:numBatches
            fprintf('    %d/%d\n',batch,numBatches)
            iter = (epoch - 1) * numBatches + batch;
            
            tic

            batchRange = (batch - 1) * batchSize + 1 : batch * batchSize;
            xs_batch = xs(:,:,batchRange);
            ys_batch = ys(:,  batchRange);

            opts.alpha = 2;
            cnn = cnnff(cnn, xs_batch);
            cnn = cnnbp(cnn, ys_batch);
            cnn = cnnapplygrads(cnn, opts);

            hist_time(iter) = toc;
            
            t_cnn = cnnff(cnn, t_xs);
            [~, Icnn] = max(t_cnn.o);
            [~, Ignd] = max(t_ys);

            hist_L(iter) = cnn.L;
            hist_o(iter,:,:) = t_cnn.o;
            %hist_Icnn(iter,:) = Icnn;
            %hist_Ignd(iter,:) = Ignd;
                       
            subplot(1,2,1)
            plot(hist_L)

            subplot(1,2,2)
            imshow(confusionmat(Ignd,Icnn)/(length(Icnn)/10))
            title(sprintf('%.3f%%',100*sum(Icnn == Ignd)/length(Icnn)))
            drawnow
        end
    end
    
    history.testPerm = testPerm;
    history.numEpochs = numEpochs;
    history.numBatches = numBatches;
    history.batchSize = batchSize;
    history.hist_L = hist_L;
    history.hist_o = hist_o;
    history.Ignd = Ignd;
    history.hist_time = hist_time;

    save(sprintf('training_%d.mat',batchSize), 'history');
end
%}

