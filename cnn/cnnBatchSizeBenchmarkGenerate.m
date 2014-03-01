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
batchSizes = fliplr([2 3 5 8 12 20 32 52 84 125 200 300 450 700 1000]);
batchSizes = batchSizes(1:5);% 6:10,11:12,13,14,15

h = waitbar(0,'Initializing waitbar...');

for bs = 1:length(batchSizes)
    batchSize = batchSizes(bs);
    numBatches = floor(numTrainingExamples / batchSize);
    numEpochs = ceil(batchSize / min(batchSizes));
    
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
        epochOrdering = randperm(60000);
        for batch = 1:numBatches
            fprintf('    %d/%d\n',batch,numBatches)
            iter = (epoch - 1) * numBatches + batch;
            
            tic

            batchRange = (batch - 1) * batchSize + 1 : batch * batchSize;
            xs_batch = xs(:,:,epochOrdering(batchRange));
            ys_batch = ys(:,  epochOrdering(batchRange));

            opts.alpha = 2;
            cnn = cnnff(cnn, xs_batch);
            cnn = cnnbp(cnn, ys_batch);
            cnn = cnnapplygrads(cnn, opts);

            hist_time(iter) = toc;
            
            totalProgress = (batch + (epoch - 1) * numBatches + (bs - 1) * numBatches * numEpochs) / (length(batchSizes) * numEpochs * numBatches);
            runtime = sum(hist_time(1:iter));
            eta = runtime / totalProgress - runtime;
            etaHours = floor(eta / 3600);
            etaMinutes = floor((eta - 3600*etaHours) / 60);
            etaSeconds = round(eta - 3600*etaHours - 60*etaMinutes);
            waitbar(totalProgress,h,sprintf('%.2f%% computed.\nTime remaining: %dh%2dm%2ds.',100*totalProgress, etaHours, etaMinutes, etaSeconds))
            
            t_cnn = cnnff(cnn, t_xs);
            [~, Icnn] = max(t_cnn.o);
            [~, Ignd] = max(t_ys);

            hist_L(iter) = cnn.L;
            hist_o(iter,:,:) = t_cnn.o;
            %hist_Icnn(iter,:) = Icnn;
            %hist_Ignd(iter,:) = Ignd;
                       
            subplot(1,2,1)
            plot(hist_L)
            title(sprintf('Batch size: %d, Batch: %d/%d, Epoch: %d/%d',batchSize,batch,numBatches,epoch,numEpochs))

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
close(h)

