function [f, fp] = parametricDynamicSystem(f, fp, t)

f = f + fp;
fp = ;

%{
t = 0:0.01:2*pi;
xt = cos(t);
yt = sin(t);

plot(xt,yt)
%}
