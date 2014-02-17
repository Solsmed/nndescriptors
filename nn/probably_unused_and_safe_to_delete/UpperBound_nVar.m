% 	
% Ch 5: Numerical Techniques - 1 D optimization
% Optimzation with MATLAB, Section 5.4.1
% Generic Scanning Procedure - n Variables
% copyright Dr. P.Venkataraman
%	
% An m-file to bracket the minimum  
% of a function of a single 
% Lower bound is known 
% only upper bound is found

% This procedure will be used along with
% Polynomial Approximation or with the Golden Section Method
%
% the following information are passed to the function
% the name of the function 			'functname'
% the function should be available as a function m.file
% and shoukd return the value of the function for a design vector

% the current position vector 			x
% the current search direction vector	s
% the initial step							a0
% the incremental step						da
% the number of bracketting steps		ns
%

%
%	sample callng statement

%  UpperBound_nVar('Example5_2',[0 0 0],[0 0 6],0,.1,10)
% 


function ReturnValue = UpperBound_nVar(functname,x,s,a0,da,ns)

format compact
%	ntrials are used to bisect/double values of da
if (ns ~= 0) ntrials = ns;
else ntrials = 10;   % default
end

if (da ~= 0) das = da;
else das = 1;  %default
end
% finds a value of function greater than or equal
% to the previous lower value

for i = 1:ntrials;
   j = 0;	dela = j*das;	a00 = a0 + dela;  
   dx0 = a00*s;	x0 = x + dx0;  f0 = feval(functname,x0);
   j = j+1;	dela = j*das;	a01 = a0 + dela;
   dx1 = a01*s;	x1 = x + dx1;	f1 = feval(functname,x1);
   f1s = f1;
   if f1 < f0 
         for j = 2:ntrials
         	a01 = a0 + j*das;		dx1 = a01*s;	
            x1 = x + dx1;		f1 = feval(functname,x1);
            f1s = min(f1s,f1);
            if f1 > f1s 
      			ReturnValue = [a01 f1 x1];
               return;
            end
         end
         fprintf('\nCannot increase function in ntrials')
      	ReturnValue = [a01 f1 x1];
         return;
			         
   else	f1 >= f0;
      das = 0.5*das;
   end
end
%fprintf('\n returned after ntrials - check problem\n')
ReturnValue =[a0 f0 x0];
   
   