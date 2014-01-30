clc; clear
load('../DeepLearnToolbox/data/mnist_uint8.mat')
original_data_x = double(train_x);
original_data_y = double(train_y);

%% Fives
yesses = 10;
noes = 10;

fives = (original_data_y(:,6) == 1);
fives = find(fives);
fives_x = original_data_x(fives(1:yesses),:);
fives_y = original_data_y(fives(1:yesses),6);

not_fives = (original_data_y(:,6) == 0);
not_fives = find(not_fives);
not_fives_x = original_data_x(not_fives(1:noes),:);
not_fives_y = original_data_y(not_fives(1:noes),6);

data_x = [fives_x; not_fives_x];
data_y = [fives_y; not_fives_y];

%% 1000
data_x = original_data_x(1:1000,:);
data_y = original_data_y(1:1000,:);

%% Handmade
data_x = [1 0 1 0 0 0 0 0;
          1 1 0 1 0 0 0 0;
          1 1 0 0 1 1 1 1]';
      
data_y =  [0 0 0 0 1 1 1 1;
           1 1 0 1 0 0 0 0]';

%% Resize input
m = 5;
method = {'nearest','triangle','cubic','lanczos2','lanczos3'};
res = 14;
data_x_small = zeros(size(data_x,1),res*res);
for i=1:size(data_x,1)
    patch = reshape(data_x(i,:),28,28);
    shrunk = imresize(patch,[res res],method{m});
    data_x_small(i,:) = shrunk(:);
end

num = randi([1 size(data_x,1)]);
subplot(1,2,1)
image(reshape(data_x(num,:),[28 28])')
title('Original MNIST')
subplot(1,2,2)
image(reshape(data_x_small(num,:),[res res])')
title(method{m})
colormap(gray(256))

data_x = data_x_small;

%% Train network

architecture = [size(data_x,2) 30 size(data_y,2)];
nn = trainNeuralNetwork(data_x,data_y,architecture);

%neuralFeedForward(nn, [0 0 1])
%%
ex = 7018;
%{
m = 5;
method = {'nearest','triangle','cubic','lanczos2','lanczos3'};
res = 14;
patch = reshape(original_data_x(ex,:),28,28);
shrunk = imresize(patch,[res res],method{m});
ex_small = shrunk(:);
image(reshape(ex_small,[res res])')
%}

tester = data_x(20,:);
%tester = original_data_x(ex,:);

%[~, i] = max(neuralFeedForward(nn,tester));
%digit = i - 1
fprintf('%.1f%% fiver\n',100*neuralFeedForward(nn,tester))
image(reshape(tester,[28 28])')