% Illustration of cost function for siamese network

Q = 256;

t = 0;
for Ew=0:Q
Lg = (2/Q).*Ew.^2;
Li = 2*Q.*exp(-2.77/Q * Ew);
Ld(Ew+1) = (t).*Lg + (1-t).*Li;
end

t = 1;
for Ew=0:Q
Lg = (2/Q).*Ew.^2;
Li = 2*Q.*exp(-2.77/Q * Ew);
Ls(Ew+1) = (t).*Lg + (1-t).*Li;
end

figure(3)
plot(0:Q, Ls, 0:Q, Ld), hold on
ylabel('Cost')
xlabel('Distance')
title(sprintf('Cost L\nBlue: similar examples\nGreen: dissimilar examples'))
xlim([0 256])

t = 0;
for Ew=0:Q
dLdEwd(Ew+1) = (t)*4/Q.*Ew - (1-t)*5.54.*exp(-2.77/Q * Ew);
end

t = 1;
for Ew=0:Q
dLdEws(Ew+1) = (t)*4/Q.*Ew - (1-t)*5.54.*exp(-2.77/Q * Ew);
end

figure(4)
plot(0:Q, dLdEws, 0:Q, dLdEwd), hold on
ylabel('Cost')
xlabel('Distance')
title(sprintf('dL / dEw\nBlue: similar examples\nGreen: dissimilar examples'))
xlim([0 256])