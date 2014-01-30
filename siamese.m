%% Load dataset from scratch
clc
clear
datasetPath = '/Volumes/Solar Flare/liberty/';
%datasetPath = 'liberty/';

patchSize = 64;

fid = fopen([datasetPath 'info.txt'], 'r');
IDs = fscanf(fid, '%d %d',[2 Inf]);
fclose(fid);

IDs = IDs(1,:)';

numEx = '10000';
fid = fopen([datasetPath sprintf('m50_%s_%s_0.txt',numEx,numEx)], 'r');
matchData = fscanf(fid, '%d %d %d %d %d %d %d',[7 Inf]);
matchData = matchData';
fclose(fid);

% clean unused columns
matchData(:,[3 6 7]) = [];
% matchData now has the form patchID point3D patchID point3D

positiveIDs = matchData(matchData(:,2) == matchData(:,4),[1 3]);
negativeIDs = matchData(matchData(:,2) ~= matchData(:,4),[1 3]);

patchIDList = nan(size(positiveIDs,1)+size(negativeIDs,1),3);
patchIDList(1:size(positiveIDs,1),:) = [positiveIDs ones(size(positiveIDs,1),1)];
patchIDList(size(positiveIDs,1)+1:end,:) = [negativeIDs zeros(size(negativeIDs,1),1)];
numPairs = size(patchIDList,1);

patches = mod(patchIDList(:,[1 2]),256);
images = floor(patchIDList(:,[1 2])/256);

patchList = nan(size(patchIDList,1), patchSize^2 + patchSize^2 + 1);

h = waitbar(0,'Initializing waitbar...');
% read image data from list of IDs
for i=1:numPairs
    i1 = imread([datasetPath 'patches' sprintf('%04d',images(i,1)) '.bmp']);
    i2 = imread([datasetPath 'patches' sprintf('%04d',images(i,2)) '.bmp']);
    
    cc = mod(patches(i,1),16)+1;
    rc = floor(patches(i,1)/16)+1;
    rows = (rc-1)*patchSize+((1:patchSize)-1)+1;
    cols = (cc-1)*patchSize+((1:patchSize)-1)+1;
    p1 = i1(rows,cols);
    
    cc = mod(patches(i,2),16)+1;
    rc = floor(patches(i,2)/16)+1;
    rows = (rc-1)*patchSize+((1:patchSize)-1)+1;
    cols = (cc-1)*patchSize+((1:patchSize)-1)+1;
    p2 = i2(rows,cols);
    
    %image([p1 p2])
    %{
    subplot(1,2,1)
    image(reshape(p1,[patchSize patchSize]))
    subplot(1,2,2)
    image(reshape(p2,[patchSize patchSize]))
    if(patchIDList(i,3))
        bgcol = [0 1 0];
    else
        bgcol = [1 0 0];
    end
    set(gcf,'Color',bgcol)
    colormap(gray(256))
    %}
    
    patchList(i,:) = [p1(:)' p2(:)' patchIDList(i,3)];
    
    waitbar(i/numPairs,h,sprintf('%.0f%% of patches loaded...',100*i/numPairs))
end
close(h)

save(sprintf('m50_%s_%s_0.mat',numEx,numEx), patchList, patchSize)
%% Load pre-loaded dataset
clear
load m50_1000_1000_0

patchesLeft = 1:patchSize^2;
patchesRight = patchSize^2 + (1:patchSize^2);
patchesSimilarity = size(patchList,2);

%%{
smallPatchSize = 16;
smallPatchList = nan(2 * smallPatchSize^2 + 1);
smallPatchesLeft = 1:smallPatchSize^2;
smallPatchesRight = smallPatchSize^2 + (1:smallPatchSize^2);
smallPatchesSimilarity = size(smallPatchList,2);
for i=1:size(patchList,1)
    p1 = patchList(i,patchesLeft);
    p2 = patchList(i,patchesRight);
    
    p1 = reshape(p1, [patchSize patchSize]);
    p2 = reshape(p2, [patchSize patchSize]);
    
    p1 = imresize(p1, [smallPatchSize smallPatchSize]);
    p2 = imresize(p2, [smallPatchSize smallPatchSize]);
    
    smallPatchList(i,smallPatchesLeft) = p1(:)';
    smallPatchList(i,smallPatchesRight) = p2(:)';
    smallPatchList(i,smallPatchesSimilarity) = patchList(i,patchesSimilarity);
end
patchesLeft = smallPatchesLeft;
patchesRight = smallPatchesRight;
patchesSimilarity = smallPatchesSimilarity;
patchSize = smallPatchSize;
patchList = smallPatchList;
%}
%% Generated data 1D - Sinusodials
patchSize = 6;
numInputs = patchSize^2;
numPairs = 1000;

patchList = nan(numPairs, 2*patchSize^2 + 1);
patchesLeft = 1:patchSize^2;
patchesRight = patchSize^2 + (1:patchSize^2);
patchesSimilarity = size(patchList,2);

patchList(:,end) = randi(2,[numPairs 1]) - 1;

for m=1:numPairs
    gen_f = rand * 2 + 0.5; % [0.5..2.5]
    gen_a = rand / 2 + 0.5; % [0.5..1.0]

    if (patchList(m,patchesSimilarity))
        gen_var = 0.1;
    else
        gen_var = 0.5;
    end
    
    patchesWhat = {patchesLeft, patchesRight};
    for gen_p=1:2
        patchList(m,patchesWhat{gen_p}) = 1/sqrt(2) * ...
            (gen_a + random('norm',0,gen_var)) * ...
            sin((gen_f + random('norm',0,gen_var)) * ...
            2*pi*(1:numInputs)/numInputs) + ...
            random('norm',0,gen_var,[1 numInputs]);
    end
    
    patchList(m,patchList(m,[patchesLeft patchesRight]) > 1) = 1;
    patchList(m,patchList(m,[patchesLeft patchesRight]) < -1) = -1;
    patchList(m,[patchesLeft patchesRight]) = patchList(m,[patchesLeft patchesRight]) / 2 + 0.5;
end
%% Look at 1D data example
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

%% Generated data 2D - Blobs and dashes
patchSize = 6;
numInputs = patchSize^2;
numPairs = 1000;

patchList = nan(numPairs, 2*patchSize^2 + 1);
patchesLeft = 1:patchSize^2;
patchesRight = patchSize^2 + (1:patchSize^2);
patchesSimilarity = size(patchList,2);

patchList(:,end) = randi(2,[numPairs 1]) - 1;

for m=1:numPairs
    gen_x = 1/(2*patchSize) + (1-1/(2*patchSize)) * rand([2+3*2 1]);
    gen_y = 1/(2*patchSize) + (1-1/(2*patchSize)) * rand([2+3*2 1]);
    gen_s = 0.1*rand([2 1]);

    if (patchList(m,patchesSimilarity))
        gen_var = 0.05;
    else
        gen_var = 0.5;
    end
    
    lsp = linspace(0+1/(2*patchSize),1-1/(2*patchSize),patchSize);
    [pX, pY] = meshgrid(lsp, lsp);
    patchesWhat = {patchesLeft, patchesRight};
    
    for gen_p=1:2
        gen_x_var = random('norm',0,gen_var,size(gen_x));
        gen_y_var = random('norm',0,gen_var,size(gen_y));
        gen_s_var = random('norm',0,gen_var,size(gen_s));
        
        p = zeros(patchSize, patchSize);
                
        linRes = 20;
        x = linspace(gen_x(3)+gen_x_var(3),gen_x(4)+gen_x_var(4),linRes);  %# x values at a higher resolution
        y = linspace(gen_y(3)+gen_x_var(3),gen_y(4)+gen_y_var(4),linRes);  %# corresponding y values
        x = [x linspace(gen_x(5)+gen_x_var(5),gen_x(6)+gen_y_var(6),linRes)];
        y = [y linspace(gen_y(5)+gen_x_var(5),gen_y(6)+gen_y_var(6),linRes)];
        x = [x linspace(gen_x(7)+gen_x_var(7),gen_x(8)+gen_y_var(8),linRes)];
        y = [y linspace(gen_y(7)+gen_x_var(7),gen_y(8)+gen_y_var(8),linRes)];
        
        x(x > 1) = 1;
        x(x < 1/(2*patchSize)) = 1/(2*patchSize);
        y(y > 1) = 1;
        y(y < 1/(2*patchSize)) = 1/(2*patchSize);
        
        index = sub2ind(size(p),round(patchSize*y),round(patchSize*x));  %# Indices of the line
        p(index) = 0.4;
        
        h = fspecial('gaussian',2,1);  %# Create filter
        p = p + filter2(h,p,'same');   %# Filter image
        
        p = p + 0.3*exp(-((gen_x(1)+gen_x_var(1) - pX).^2 + (gen_y(1)+gen_y_var(1) - pY).^2)/(2*abs(gen_s(1)+gen_s_var(1))));
        p = p + 0.3*exp(-((gen_x(2)+gen_x_var(2) - pX).^2 + (gen_y(2)+gen_y_var(2) - pY).^2)/(2*abs(gen_s(2)+gen_s_var(2))));

        %imshow(p)
       
        patchList(m,patchesWhat{gen_p}) = p(:);
    end
    
%    patchList(m,patchList(m,[patchesLeft patchesRight]) > 1) = 1;
%    patchList(m,patchList(m,[patchesLeft patchesRight]) < -1) = -1;
%    patchList(m,[patchesLeft patchesRight]) = patchList(m,[patchesLeft patchesRight]) / 2 + 0.5;
end
    
%% Look at 2D data example
s = 213;
subplot(1,2,1)
image(256*reshape(patchList(s,patchesLeft),[patchSize patchSize]))
subplot(1,2,2)
image(256*reshape(patchList(s,patchesRight),[patchSize patchSize]))
if(patchList(s,patchesSimilarity))
    bgcol = [0 1 0];
else
    bgcol = [1 0 0];
end
set(gcf,'Color',bgcol)
colormap(gray(256))

%% Train
addpath('../DeepLearnToolbox/util')
addpath('../DeepLearnToolbox/NN')
numOutputs = 10;

trainNum = 1;

seed = randperm(size(patchList,1));
trainSet = seed(1:trainNum);
testSet = seed((trainNum + 1):end);
x1 = patchList(trainSet,patchesLeft);
x2 = patchList(trainSet,patchesRight);
t = patchList(trainSet,patchesSimilarity);

architecture = [patchSize^2 numOutputs];

nn1 = nnsetup(architecture);
nn2 = nnsetup(architecture);
nn1.activation_function = 'sigm';
nn2.activation_function = 'sigm';
nn1.W = nn2.W;
numInterfaces = nn1.n - 1;

numIters = 100;
hist_L = nan(numIters, 1);

for iter=1:numIters
    fprintf('iter %d ',iter)
    
    %{
    s = 1;
    subplot(1,2,1)
    image(reshape(x1(s,:),[patchSize patchSize]))
    subplot(1,2,2)
    image(reshape(x2(s,:),[patchSize patchSize]))
    if(t(s))
        bgcol = [0 1 0];
    else
        bgcol = [1 0 0];
    end
    set(gcf,'Color',bgcol)
    %}
    
    %
    %   F O R W A R D   P A S S
    %
    
    % Feed forward NN twins
    nn1 = nnff(nn1, x1, 0);
    nn2 = nnff(nn2, x2, 0);
    fprintf('.')
    
    % Each twin's ouput vector
    g1 = nn1.a{nn1.n};
    g2 = nn2.a{nn2.n};
    
    % Siamese output
    % Descriptor distance, L1 norm, normalised
    Ew_fun = @(g1_fun, g2_fun) sum(abs((g1_fun - g2_fun)),2);
    Ew = Ew_fun(g1, g2);
    
    %Traditional logarithmic cost function
    %L = sum( (1-t).*log(1-Ew) + t.*log(Ew) );
    % Cost function, LeCun's L
    % Notre Damme labels are inversed compared to LeCun's formula
    % LeCun: Y = 1 <=> impostor, Y = 0 <=> genuine
    % ND:    t = 1 <=> similar, t = 0 <=> dissimilar
    Q = numOutputs; % upper bound for Ew
    
    %Lg = (2/Q).*Ew.^2;
    %Li = 2*Q.*exp(-2.77/Q * Ew);
    %L = (t).*Lg + (1-t).*Li;
    %L = sum(L) / trainNum;
    L=@(Ew, t) ((t).*(2/Q).*Ew.^2 + (1-t).*2*Q.*exp(-2.77/Q * Ew));
    hist_L(iter) = mean(L(Ew, t));
    nn1.L = L(Ew, t);
    nn2.L = L(Ew, t);
    
    
    %
    %   B A C K P R O P A G A T I O N   (D E R I V A T I V E S)
    %
    epsilon = 1e-7;
%%    
    % dL/dEw
    %
    % analytical
    dLdEw_fun = @(Ew_val, t_val) ((t_val).*4/Q.*Ew_val - (1-t_val).*5.54.*exp(-2.77/Q * Ew_val)); % scalar
    dLdEw = dLdEw_fun(Ew, t);
    %dLdEw = (1-t).*5.54.*exp(-2.77/Q * Ew) / trainNum; % scalar
    % numerical [VERIFIED TO BE SAME AS ANALYTICAL!]
    %dLdEwng = (L(Ew + epsilon, t) - L(Ew - epsilon, t))/(2*epsilon);
    
    % dEw/dG
    %
    % analytical
    dEwdG_fun = @(g1_val, g2_val) sign(g1_val - g2_val);
    dEwdG = dEwdG_fun(g1, g2);
    % numerical [VERIFIED TO BE SAME AS ANALYTICAL!]
    %{
    dEwdGng = nan(trainNum, numOutputs);
    %dEwdG1ng = nan(trainNum, numOutputs);
    %dEwdG2ng = nan(trainNum, numOutputs);
    for ind = 1:numOutputs
        epsilonVec = zeros(1, numOutputs);
        epsilonVec(ind) = epsilon;
        epsilonVec = repmat(epsilonVec, [trainNum 1]);
        %dEwdG1ng(:,ind) = (Ew_fun(g1 + epsilonVec, g2) - Ew_fun(g1 - epsilonVec, g2)) / (2*epsilon);
        %dEwdG2ng(:,ind) = (Ew_fun(g1, g2 + epsilonVec) - Ew_fun(g1, g2 - epsilonVec)) / (2*epsilon);
        dEwdG1ng = (Ew_fun(g1 + epsilonVec, g2) - Ew_fun(g1 - epsilonVec, g2)) / (2*epsilon);
        dEwdG2ng = (Ew_fun(g1, g2 + epsilonVec) - Ew_fun(g1, g2 - epsilonVec)) / (2*epsilon);
        dEwdGng(:,ind) = (dEwdG1ng + (-dEwdG2ng)) / 2;
    end
    %}
    
    % dGdW
    %
    % algorithmical
    %%{
    dGdW = cell(size(nn1.W));
    G1error =  repmat(dLdEw,[1 numOutputs]) .* dEwdG; 
    G2error = -repmat(dLdEw,[1 numOutputs]) .* dEwdG;
    nn1.e = G1error;
    nn2.e = G2error;
    nn1 = nnbp(nn1);
    nn2 = nnbp(nn2);
    dG1dW = nn1.dW;
    dG2dW = nn2.dW;
    for layer=1:numInterfaces
        dGdW{layer} = (dG1dW{layer} + dG2dW{layer}) / 2;
    end
    %}
    % numerical
    %%{
    %%
    dG1dWng = cell(size(nn1.W));
    dG2dWng = cell(size(nn2.W));
    for layer=1:numInterfaces
        dG1dWng{layer} = nan([size(nn1.W{layer}) trainNum numOutputs]);
        dG2dWng{layer} = nan([size(nn2.W{layer}) trainNum numOutputs]);
    end
    for layer=1:numInterfaces
        for i=1:size(dG1dWng{layer}, 1)
            for j=1:size(dG1dWng{layer}, 2)
                wij = nn1.W{layer}(i,j);
                
                % Do nudging of wij both upwards and downwards
                dG1dWngij = nan(2, trainNum, numOutputs);
                dG2dWngij = nan(2, trainNum, numOutputs);
                for s=[-1 1]
                    nn1.W{layer}(i,j) = wij + s*epsilon;
                    nn2.W{layer}(i,j) = wij + s*epsilon;
                    
                    % Save the response of G1 and G2 at each nudge
                    nn1ng = nnff(nn1, x1, 0);
                    nn2ng = nnff(nn2, x2, 0);

                    dG1dWngij((s+1)/2 + 1,:,:) = nn1ng.a{end};
                    dG2dWngij((s+1)/2 + 1,:,:) = nn2ng.a{end};
                end
                
                dG1dWng{layer}(i,j,:,:) = (dG1dWngij(2,:,:) - dG1dWngij(1,:,:)) / (2*epsilon);
                dG2dWng{layer}(i,j,:,:) = (dG2dWngij(2,:,:) - dG2dWngij(1,:,:)) / (2*epsilon);
                
                nn1.W{layer}(i,j) = wij;
            end
        end
    end
    dGdWng = cell(size(dG1dWng));
    for layer=1:length(dG1dWng)
        dGdWng{layer} = (dG1dWng{layer} + dG2dWng{layer}) / 2;
    end
    %}
    %%
    % Compute dLdW (using some mix of derivatives)
    %{
    dLdW = cell(size(nn1.W));
    for layer=1:numInterfaces
        ws1 = size(nn1.W{layer},1);
        ws2 = size(nn1.W{layer},2);
        dLdW{layer} = zeros([ws1 ws2 trainNum]);
        for i=1:ws1
            for j=1:ws2
                dLdW{layer}(i,j,:) = dLdW{layer}(i,j,:) + reshape(...
                    sum(...
                    reshape(dGdWng{layer}(i,j,:,:),[trainNum numOutputs]) .* ...
                    dEwdG,2) .* ...
                    dLdEw, ...
                    [1 1 100]);
            end
        end
        %dLdW{layer} = dLdW{layer} / trainNum;
    end
    % Set dW
    for layer=1:numInterfaces
        nn1.dW{layer} = mean(dLdW{layer},3);
        nn2.dW{layer} = mean(dLdW{layer},3);
    end
    %}

    %         dL/dW
    % Numerical, whole system (last resort)
    %%{
    dLdWng = cell(size(nn1.W));
    for layer=1:numInterfaces
        dLdWng{layer} = nan([size(nn1.W{layer}) trainNum]);
    end  
    for layer=1:length(dLdWng)
        %fprintf('Layer %d\n',layer);
        ws1 = size(nn1.W{layer},1);
        ws2 = size(nn1.W{layer},2);
        for i=1:ws1
            %fprintf('%d/%d\n',i,size(dLdWng{layer},1));
            for j=1:ws2
                wij = nn1.W{layer}(i,j);
                
                Le = nan(2,trainNum);
                for s=[-1 1]
                    nn1.W{layer}(i,j) = wij + s*epsilon;
                    nn2.W{layer}(i,j) = wij + s*epsilon;

                    nn1ng = nnff(nn1, x1, 0);
                    nn2ng = nnff(nn2, x2, 0);
                    g1ng = nn1ng.a{end};
                    g2ng = nn2ng.a{end};
                    Le((s+1)/2 + 1,:) = L(Ew_fun(g1ng, g2ng), t);
                end

                dLdWng{layer}(i,j,:) = (Le(2,:) - Le(1,:)) / (2*epsilon);

                nn1.W{layer}(i,j) = wij;
                nn2.W{layer}(i,j) = wij;
            end
        end
    end
    
    %}
    
    %
    %   N U M E R I C A L   T R I A L  &  E R R O R   S E C T I O N
    %
    % Shows, no proofs
    %
    % This is to show that it's correct to use repmat and .*
    % when chaining derivatives together
    %
    %{
    dLdG = repmat(dLdEw, [1 numOutputs]) .* dEwdG;
    % SHOW HERE, SAME RESULT WHEN DOING NUMERICAL DIFFERENTIATION
    %{
    dLdGng = nan(trainNum, numOutputs);
    dLdG1ng = nan(trainNum, numOutputs);
    dLdG2ng = nan(trainNum, numOutputs);
    for ind = 1:numOutputs
        epsilonVec = zeros(1, numOutputs);
        epsilonVec(ind) = epsilon;
        epsilonVec = repmat(epsilonVec, [trainNum 1]);
        dLdG1ng(:,ind) = (L(Ew_fun(g1 + epsilonVec, g2),t) - L(Ew_fun(g1 - epsilonVec, g2),t)) / (2*epsilon);
        dLdG2ng(:,ind) = (L(Ew_fun(g1, g2 + epsilonVec),t) - L(Ew_fun(g1, g2 - epsilonVec),t)) / (2*epsilon);
        %dLdG1ng = (L(Ew_fun(g1 + epsilonVec, g2),t) - L(Ew_fun(g1 - epsilonVec, g2),t)) / (2*epsilon);
        %dLdG2ng = (L(Ew_fun(g1, g2 + epsilonVec),t) - L(Ew_fun(g1, g2 - epsilonVec),t)) / (2*epsilon);
        %dLdGng(:,ind) = (dLdG1ng + -dLdG2ng) / 2;
    end
    
    shouldBeZero = sum(sum(abs(dLdG1ng))) - sum(sum(abs(dLdG2ng))); % same values (expr = zero)
    shouldBeZero = sum(sum(abs(dLdG1ng + dLdG2ng))); % same values, opposite signs (expr = zero)
    
    dLdGng = (dLdG1ng + -dLdG2ng) / 2;
    %}
    %}    
    
    %         dEw/dW
    % Numerical
    %{
    dEwdWng = cell(size(nn1.W));
    for layer=1:numInterfaces
        dEwdWng{layer} = nan([size(nn1.W{layer}) trainNum]);
    end  
    for layer=1:length(dEwdWng)
        %fprintf('Layer %d\n',layer);
        ws1 = size(nn1.W{layer},1);
        ws2 = size(nn1.W{layer},2);
        for i=1:ws1
            %fprintf('%d/%d\n',i,size(dLdWng{layer},1));
            for j=1:ws2
                wij = nn1.W{layer}(i,j);
                
                Le = nan(2,trainNum);
                for s=[-1 1]
                    nn1.W{layer}(i,j) = wij + s*epsilon;
                    nn2.W{layer}(i,j) = wij + s*epsilon;

                    nn1ng = nnff(nn1, x1, 0);
                    nn2ng = nnff(nn2, x2, 0);
                    g1ng = nn1ng.a{end};
                    g2ng = nn2ng.a{end};
                    Le((s+1)/2 + 1,:) = Ew_fun(g1ng, g2ng);
                end

                dEwdWng{layer}(i,j,:) = (Le(2,:) - Le(1,:)) / (2*epsilon);

                nn1.W{layer}(i,j) = wij;
                nn2.W{layer}(i,j) = wij;
            end
        end
    end
    
    % dLdW / dEwdW = (dLdEw * dEwdW) / dEwdW = dLdEw
    dLdEwng2 = dLdWng{1} ./ dEwdWng{1};
    % dLdEw / dLdEw = 1
    shouldBeOnes = dLdEwng2 ./ repmat(reshape(dLdEw,[1 1 100]),[10 37 1]);
    closeToZero = sum(abs((shouldBeOnes(:) - 1))) / numel(shouldBeOnes);
    %}
    
    % Try to find dGdW using the fact that dLdW / (dLdEw * dEwdG) = dGdW
    
    dLdG = (repmat(dLdEw,[1 numOutputs]) .* dEwdG);
    
    % for example m
    % dGdWng{layer}(i,j,m,1:numOutputs) is the way wij alters output vector
    % dLdG(m,:) is the way output vector's components affect cost
    % so we want to find the combination of wij modifications that maximise
    % the resemblence 
   
   %%  
    % Naive way of separating output vectors
    %{
    direction = repmat(sign(t - 0.5),[1 numOutputs]); % 1 = sim, -1 = dis
    g2Tog1 = (g1 - g2);
    g1Tog2 = -g2Tog1;
    G1error =  1/2 * g1Tog2 .* direction;
    G2error =  1/2 * g2Tog1 .* direction;
    %}

  
    fprintf('.')
  

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
    xlim([1 numIters])
    ylim([0 10])
    
    subplot(2,2,3)
    [threshold, truePos, trueNeg, falsePos, falseNeg] = testSiamese(nn1, @(a, b) Ew_fun(a, b)/numOutputs, testSet, patchList);
    
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
    %%
    % Set dW
    for layer=1:numInterfaces
        %nn1.dW{layer} = dGdW{layer};
        %nn2.dW{layer} = dGdW{layer};
        
        nn1.dW{layer} = mean(dLdWng{layer},3);
        nn2.dW{layer} = mean(dLdWng{layer},3);
    end
    
    nn1p = nnapplygrads(nn1);
    nn2p = nnapplygrads(nn2);
    nn1p = nnff(nn1p, x1, 0);
    nn2p = nnff(nn2p, x2, 0);
    g1p = nn1p.a{end};
    g2p = nn2p.a{end};
    fprintf('!')
    fprintf(' Cost %.5f',hist_L(iter))
    fprintf('\n')
end