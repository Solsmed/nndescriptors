function [ patchList, patchSize, patchesLeft, patchesRight, patchesSimilarity ] = loadData( type, nameNumEx, loadPath, newSize)
%loadData Returns some (un)matching-pair data to learn.
%
%   Arguments: type, nameNumEx, loadPath, newSize
%
%   type - 'liberty':
%   nameNumEx - 1000 or 10000 etc
%   loadPath - relative path to dataset files
%   newSize - patchSize, this function can resize data
%
%   type - 'generate1D'
%   nameNumEx - number of examples
%   loadPath not used
%   newSize - sqrt(size of data)
%
%   type - 'generate2D'
%   nameNumEx - number of examples
%   loadPath not used
%   newSize - size of data
%
%   type - 'load'
%   nameNumEx - (unsused)
%   loadPath - filename to load
%   newSize - optional, if you want to resize the loaded data

saveName = '';

patchList = NaN;
patchSize = NaN;
patchesLeft = NaN;
patchesRight = NaN;
patchesSimilarity = NaN;

switch type
    case 'liberty'
        % Load dataset from scratch
        patchSize = 64;
        
        fid = fopen([loadPath 'info.txt'], 'r');
        IDs = fscanf(fid, '%d %d',[2 Inf]);
        fclose(fid);

        IDs = IDs(1,:)';

        fid = fopen([loadPath sprintf('m50_%d_%d_0.txt',nameNumEx,nameNumEx)], 'r');
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
            i1 = imread([loadPath 'patches' sprintf('%04d',images(i,1)) '.bmp']);
            i2 = imread([loadPath 'patches' sprintf('%04d',images(i,2)) '.bmp']);

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

            p1 = double(p1)/255;
            p2 = double(p2)/255;
            
            % Realtime show as pairs are loaded
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

        saveName = sprintf('m50_%d_%d_0.mat',nameNumEx,nameNumEx);
    case 'generate1D'
        % Generate 1D data - Sinusodials
        if (nargin >= 4)
            patchSize = newSize;
        else
            patchSize = 6;
        end
       
        numInputs = patchSize^2;
        patchList = nan(nameNumEx, 2*patchSize^2 + 1);
        patchesLeft = 1:patchSize^2;
        patchesRight = patchSize^2 + (1:patchSize^2);
        patchesSimilarity = size(patchList,2);

        patchList(:,end) = randi(2,[nameNumEx 1]) - 1;

        for m=1:nameNumEx
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
    case 'generate2D'
        % Generate 2D data - Blobs and dashes
        if (nargin >= 4)
            patchSize = newSize;
        else
            patchSize = 6;
        end

        patchList = nan(nameNumEx, 2*patchSize^2 + 1);
        patchesLeft = 1:patchSize^2;
        patchesRight = patchSize^2 + (1:patchSize^2);
        patchesSimilarity = size(patchList,2);

        patchList(:,end) = randi(2,[nameNumEx 1]) - 1;

        for m=1:nameNumEx
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
        end
    case 'load'
        load(loadPath)
end

if (nargin >= 4 && newSize > 0 && newSize < patchSize)
    resizedPatchSize = newSize;
    resizedPatchList = nan(size(patchList,1), 2*resizedPatchSize^2 + 1);
    resizedPatchesLeft = 1:resizedPatchSize^2;
    resizedPatchesRight = resizedPatchSize^2 + (1:resizedPatchSize^2);
    resizedPatchesSimilarity = size(resizedPatchList,2);
    for i=1:size(patchList,1)
        p1 = patchList(i,patchesLeft);
        p2 = patchList(i,patchesRight);

        p1 = reshape(p1, [patchSize patchSize]);
        p2 = reshape(p2, [patchSize patchSize]);

        p1 = imresize(p1, [resizedPatchSize resizedPatchSize]);
        p2 = imresize(p2, [resizedPatchSize resizedPatchSize]);

        resizedPatchList(i,resizedPatchesLeft) = p1(:)';
        resizedPatchList(i,resizedPatchesRight) = p2(:)';
        resizedPatchList(i,resizedPatchesSimilarity) = patchList(i,patchesSimilarity);
    end
    patchSize = resizedPatchSize;
    patchList = resizedPatchList;
    %patchesLeft = resizedPatchesLeft;
    %patchesRight = resizedPatchesRight;
    %patchesSimilarity = resizedPatchesSimilarity;
end

patchesLeft = 1:patchSize^2;
patchesRight = patchSize^2 + (1:patchSize^2);
patchesSimilarity = size(patchList,2);
    
if (~isempty(saveName))
    save(saveName,'patchList','patchSize','patchesLeft','patchesRight','patchesSimilarity');
end

end

