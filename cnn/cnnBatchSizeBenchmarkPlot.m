histories = cell(14,1);
batchSizes = [3 5 8 12 32 48 80 125 200 300 400 600 800 1000];
for i = 1:length(histories)
    load(sprintf('../data/training_%d.mat',batchSizes(i)))
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

% animate
% Each example's output has a standard deviation.
% Ideally, we want an output to have one output at 1 and the rest at 0.
% One way of summarising this is as a scalar is to measure the output's
% standard deviation.
% Plot shows the distribution of standard deviations across all examples
% over time, as the network learns and becomes more specific.
% The distribution is sorted, so that the line becomes smooth and viewable.
%
% Line close to zero is good. Each point on the line that is zero reflects
% and example which the network is very sure about (albeit not necessarily
% correct); only one output is 1, the rest are 0.
%
% (Note: this measure is somewhat arbitrary and does not perfectly fill its
%        purpose)
for t = timeFrames
    ouputs = squeeze(outputAnimation(t == timeFrames,:,:));
    stdAtFrames = sqrt(var(ouputs,0,1));
    targetStd = sqrt(var([0 0 0 0 0 0 0 0 0 1],0,2));
    plot(sort(abs(stdAtFrames - targetStd)))
    ylim([0 0.5])
    drawnow
    pause(0.4)
end

%% Plot how batch size affects accuracy in learning over time
numFrames = 30;

time = nan(numel(histories), numFrames);
accuracy = nan(numel(histories), numFrames);
batchSizes = nan(numel(histories), numFrames);

for ver = 1:numel(histories)
    fprintf('%d/%d\n',ver,numel(histories))
    
    b = 1;
    for batch = round(linspace(1,histories{ver}.numBatches, numFrames))
        fprintf('    %d/%d\n',batch,histories{ver}.numBatches)
        
        [~, Icnn] = max(squeeze(histories{ver}.hist_o(batch,:,:)));
        Ignd = histories{ver}.Ignd;
        
        time(ver, b) = sum(histories{ver}.hist_time(1:batch));
        accuracy(ver, b) = sum(Icnn == Ignd)/length(Icnn);
        b = b + 1;
    end
    
    batchSizes(ver,:) = histories{ver}.batchSize;
end


surf(log(batchSizes),time,accuracy)
xlabel('log batchSize')
ylabel('effective runtime (seconds)')
zlabel('accuracy (%)')