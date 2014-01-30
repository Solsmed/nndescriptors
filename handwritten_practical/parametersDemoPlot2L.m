%% TWO HIDDEN LAYERS
%clear
%load('nn2L_10-50,10-50'), numTests = 10000;
load('nn2L_10-100,10-60')


figure(5)
clf
e = eye(numOutputs);
Acorr = reshape(hist_A(:,end,:,:),[numOutputs^2 numStats numArchs]);
Acorr = Acorr.*repmat(e(:),[1 numStats numArchs]);
Acorr = reshape(sum(Acorr,1),[numStats numArchs]);
Acorr = 100*Acorr / numTests;
L1len = length(nc{1});
Acorr = reshape(Acorr,[numStats numArchs/L1len L1len]);
Acorr = permute(Acorr,[3 2 1]);
statplot3d(nc{1},nc{2},Acorr);
title('Correctness')
xlabel('Layer 1 neuron count')
ylabel('Layer 2 neuron count')
zlabel('Percent')
view([60 15])



figure(6)
clf
time = sum(reshape(mean(hist_iterTime,2),[numIters numArchs]),1);
efficiency = mean(Acorr,3)./reshape(time,L1len,numArchs/L1len);
surf(nc{1},nc{2},efficiency','edgecolor','none');
title('Computational efficiency')
xlabel('Layer 1 neuron count')
ylabel('Layer 2 neuron count')
zlabel('Percent change per second')
view([60 15])
colormap(jet(256))