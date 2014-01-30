function net = cnntrain(net, x, y, opts, test_x, test_y)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        for l = 1 : numbatches
            fprintf('    %d/%d\n',l,numbatches)
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            
            t_cnn = cnnff(net, test_x);
            [~, Icnn] = max(t_cnn.o);
            [~, Ignd] = max(test_y);
            
            if isempty(net.rL)
                net.rL(1) = net.L;
                net.wL(1) = net.L;
            end
            net.wL(end + 1) = net.L;
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            
            subplot(1,2,1)
            imshow(confusionmat(Ignd,Icnn)/(length(Icnn)/10))
            title(sprintf('%.3f%%',100*sum(Icnn == Ignd)/length(Icnn)))
            subplot(1,2,2)
            plot(net.wL), hold on
            plot(net.rL,'r')
            xlim([1 1200])
            ylim([0 1.5])
            drawnow
        end
        toc;
    end
    
end
