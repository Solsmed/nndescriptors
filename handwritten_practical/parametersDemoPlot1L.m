%% ONE HIDDEN LAYER - Correctness, Efficiency
%clear
load('nn_1-100')
numTests = 1000;

figure(1)
clf
Ameans = reshape((mean(hist_A,3)),[numOutputs^2 numIters numArchs]);
e = eye(numOutputs);
Acorrs=(Ameans.*repmat(e(:),[1 numIters numArchs]));
Acorrs = reshape(sum(Acorrs,1),[numIters numArchs]);
Acorrs = 100*Acorrs / numTests;
surf(1:numIters,nc,Acorrs','edgecolor','none')
title('Correctness (mean)')
xlabel('Iteration')
ylabel('Number of neurons in hidden layer')
zlabel('Percent correct')
view([60 15])
colormap(jet(256))

time = sum(reshape(mean(hist_iterTime,2),[numIters numArchs]),1);

efficiency = Acorrs./repmat(time,[numIters 1]);

figure(2)
clf
surf(1:numIters,nc,efficiency','edgecolor','none');
title('Computational efficiency')
xlabel('Iteration')
ylabel('Number of neurons in hidden layer')
zlabel('Percent change per second')
view([60 15])
colormap(jet(256))

%% ONE HIDDEN LAYER - Look at correctness in stereo (cross-eyed)
figure(3)
clf
ang = 60;
az = 15;


corrA = hist_A.*repmat(e(:),[1 numIters numStats numArchs]);
corrA = sum(corrA,1) / numTests;
corrA = reshape(corrA,[numIters numStats numArchs]);
corrA = permute(corrA,[1 3 2]);
subplot(1,2,1)
statplot3d(1:numIters, nc, corrA);
view([ang+2 az])
camproj('perspective')
%title('Correctness (25%, median, 75% quantiles)')
%xlabel('Iteration')
%ylabel('Number of neurons in hidden layer')
%zlabel('Percent change per second')

subplot(1,2,2)
statplot3d(1:numIters, nc, corrA);
view([ang-2 az])
camproj('perspective')


%% ONE HIDDEN LAYER - Look at how NN-matrix develops over time
figure(4)
clf

Aimg = reshape(hist_A(:,5,5,find(nc==15)),numOutputs,numOutputs);
Aimg = Aimg ./ repmat(sum(Aimg,2),[1 numOutputs]);
dist = norm(Aimg(:) - e(:));
roughness = 0;

for t=6:numIters
    Aimg = reshape(hist_A(:,t,5,find(nc==20)),numOutputs,numOutputs);
    Aimg = Aimg ./ repmat(sum(Aimg,2),[1 numOutputs]);
    
    subplot(2,1,1)
    image(256*mat2gray(Aimg))
    set(gca, 'color',0.8*[1 1 1])
    axis equal, axis off
    
    newDist = norm(Aimg(:) - e(:));
    roughness = roughness + (newDist - dist)^2;
    dist = norm(Aimg(:) - e(:));
    
    subplot(2,2,3)
    plot(t,dist,'r.'), hold on
    xlim([1 100])
    ylim([0 4])
    title(sprintf('Roughness %.1f',100*roughness))
    
    subplot(2,2,4)
    plot(t,100*roughness,'xk'), hold on
    xlim([1 100])
    ylim([0 20])
    
    drawnow
end