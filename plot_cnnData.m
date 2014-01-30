numFrames = 12;
bob = nan(numFrames,size(hist_o,2),size(hist_o,3));
timeFrames = floor(linspace(1,size(hist_o,1),numFrames));

% pre-load
for t = timeFrames
    bob(t == timeFrames,:,:) = squeeze(hist_o(t,:,:));
    fprintf('.')
end
fprintf('\n')

% animate
for t = timeFrames
    plot(sort(abs(sqrt(var(squeeze(bob(t == timeFrames,:,:)),0,1)) - sqrt(var([0 0 0 0 0 0 0 1 0 0],0,2)))))
    ylim([0 0.5])
    drawnow
    pause(0.4)
end

%%
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