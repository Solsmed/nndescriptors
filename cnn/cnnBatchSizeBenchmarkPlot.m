histories = cell(15,1);
%batchSizes = [3 5 8 12 32 48 80 125 200 300 400 600 800 1000];
batchSizes = fliplr([2 3 5 8 12 20 32 52 84 125 200 300 450 700 1000]);
for i = 1:length(histories)
    load(sprintf('training_%d.mat',batchSizes(i)))
    histories{i} = history;
    fprintf('loaded training_%d.mat (%d/%d)\n',batchSizes(i),i,length(histories))
end

%% Animate the networks spcificity/precision development over time
numFrames = 12;
selectedHistoryNo = 1;
hist_o = histories{selectedHistoryNo}.hist_o;

outputAnimation = nan(numFrames,size(hist_o,2),size(hist_o,3));
timeFrames = floor(linspace(1,size(hist_o,1),numFrames));

% pre-load
for t = timeFrames
    outputAnimation(t == timeFrames,:,:) = squeeze(hist_o(t,:,:));
    fprintf('.')
end
fprintf('\n')

% Ideally, we want an output to have one output node at 1 and the rest at 0
% as this means the network is very certain. To express how well the output
% vector fits this ideal shape we can look at the output vector's standard
% deviation (std of the values in its output nodes).
%
% (Note: this measure is somewhat arbitrary and does not perfectly fill its
%        purpose)
certainness = nan(length(timeFrames),1);

for t = 1:length(timeFrames)
    ouputs = squeeze(outputAnimation(t,:,:));
    stdAtFrames = sqrt(var(ouputs,0,1));
    targetStd = sqrt(var([0 0 0 0 0 0 0 0 0 1],0,2));

    % Animate
    %
    % Plot shows the distribution of standard deviations for all examples
    % at the current time, as the network learns and becomes more specific.
    % The distribution is sorted, so that the line becomes smooth and
    % viewable.
    %
    % Line close to zero is good. Each point on the line that is zero
    % reflects and example which the network is very sure about (albeit not
    % necessarily correct); only one output is 1, the rest are 0.
    %
    figure(1)
    plot(sort(abs(stdAtFrames - targetStd)))
    ylim([0 0.5])
    title(sprintf('Deviation from ideal output vector after %d iterations', timeFrames(t)))
    xlabel('Example # (sorted by deviation)')
    ylabel('Deviation')
    drawnow
    pause(0.4)
    
    % Summary of certainness across all examples efter time t.
    certainness(t) = mean(abs(stdAtFrames - targetStd));
end

figure(2)
plot(timeFrames,certainness)
title('Development of network uncertainty')
xlabel('Iterations')
ylabel('Mean deviation from ideal certainty')
%% Plot how batch size affects accuracy in learning over time
numFrames = 30;

iterationtime = nan(numel(histories), numFrames);
realtime = nan(numel(histories), numFrames);
accuracy = nan(numel(histories), numFrames);
batchSizes = nan(numel(histories), numFrames);

for ver = 1:numel(histories)
    fprintf('%d/%d\n',ver,numel(histories))
    
    b = 1;
    for batch = round(linspace(1,histories{ver}.numBatches*histories{ver}.numEpochs, numFrames))
        fprintf('    %d/%d\n',batch,histories{ver}.numBatches)
        
        [~, Icnn] = max(squeeze(histories{ver}.hist_o(batch,:,:)));
        Ignd = histories{ver}.Ignd;
        
        iterationtime(ver, b) = batch;
        realtime(ver, b) = sum(histories{ver}.hist_time(1:batch));
        
        accuracy(ver, b) = sum(Icnn == Ignd)/length(Icnn);
        b = b + 1;
    end
    
    batchSizes(ver,:) = histories{ver}.batchSize;
end

logAccuracy = -log10(1 - accuracy);

figure(3)
surf(log10(batchSizes),iterationtime,logAccuracy)
xlabel('log10 batchSize')
ylabel('runtime (iterations)')
zlabel('log10 accuracy (%)')

figure(4)
surf(log10(batchSizes),log10(realtime),logAccuracy)
xlabel('log10 batchSize')
ylabel('log10 runtime (seconds)')
zlabel('log10 accuracy (%)')