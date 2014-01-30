function Return = gradfunction(functname,x)
%
% numerical computation of gradient
% this allows automatic gradient computation
% 
%
% first forward finite difference
% hstep = 0.001; - programmed in
%
hstep = 0.001;
n = length(x);
f = feval(functname,x);

for i = 1:n
   xs = x;
   xs(i) = xs(i) + hstep;
   gradx(i)= (feval(functname,xs) -f)/hstep;
end
Return = gradx;