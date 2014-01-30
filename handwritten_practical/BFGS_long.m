% 	
%
% Broydon - Fletcher - Goldfarb - Shanno (BFGS) Method
%	
% An m-file for the BFGS Method
%************************************
% requires:     	      UpperBound_nVar.m
%						  GoldSection_nVar.m
% and the problem m-file: Objectivefunction.m
% 
%***************************************
%
% the following information are passed to the function

% the name of the function 			    'functname' 
% functname.m : returns scalar for vector input
%
% the gradient calculation is in        gradfunction.m
% gradfunction.m:  returns vector for vector input
%
% initial design vector					dvar0
%  number of iterations                 niter

%------for golden section
%  the tolerance (for golden section)	tol 
%
%-------for upper bound calculation
% the initial value of stepsize			lowbound
% the incremental value 				intvl
% the number of scanning steps	    	ntrials
%
% the function returns the final design and the objective function

%	sample callng statement

% BFGS('Objectivefunction',[0.5 0.5],20, 0.0001, 0,1 ,20)
%
function ReturnValue = BFGS(functname, ...
    dvar0,niter,tol,lowbound,intvl,ntrials)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% management functions
clc    % position the cursor at the top of the screen
clf   %  closes the figure window
format compact  % avoid skipping a line when writing to the command window
warning off  % don't report any warnings like divide by zero etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nBFGS Method\n');
fprintf('The problem:  '),disp(functname)

% convergence/stopping criteria
e1 = 1.0e-04; e2 = 1.0e-08; e3 = 1.0e-04;  
nvar = length(dvar0); % length of design vector or number of variables
% obtained from start vector 
if (nvar == 2)
    %*************************
    %  plotting contours -
    %  only for two variables
    %  previous generation code is left in place
    %*************************
    % the plot is centered around initial guess
    % with (+-) delx1, delx2 on either side
    % this can be reset by user

    delx1 = 6;
    delx2 = 5;

    x1 = (dvar0(1)-delx1):0.1:(dvar0(1)+delx1);
    x2 = (dvar0(2)-delx2):0.1:(dvar0(2)+delx2);
   
    x1len = length(x1);
    x2len = length(x2);
    %     [X1 X2] = meshgrid(x1,x2);
    %     Vfunctname = strcat('V',functname);
    %     fun = feval(Vfunctname,X1,X2);
    for i = 1:x1len;
        for j = 1:x2len;
            x1x2 =[x1(i) x2(j)];
            fun(j,i) = feval(functname,x1x2);
        end
    end
   
  c1 = contour(x1,x2,fun, ...
   	[3.1 3.25 3.5 4 6 10 15 20 25],'k');
    %clabel(c1); % remove labelling for clarity
   
    grid
    xlabel('x_1');
    ylabel('x_2');
    funname = strrep(functname,'_','-');
    title(strcat('BFGS:',funname));

    % note that contour values are problem dependent
    % the range is problem dependent
    %**************************
    % finished plotting contour
    %***************************

end
%*********************
%  Numerical Procedure
%*********************
% design vector, alpha , and function value is stored
xs(1,:) = dvar0;
x = dvar0;
Lc = 'r';
fs(1) = feval(functname,x); % value of function at start
as(1)=0;
grad = (gradfunction(functname,x)); % steepest descent

A = eye(nvar);  % initial metric
% uses MATLAB built in identity matrix function

convg(1)=grad*grad';
for i = 1:niter-1
    % determine search direction
    fprintf('iteration number:  '),disp(i)
    %s = (-A*grad')'; % s is used as a row vector
    s = (-inv(A)*grad')';
    
    output = GoldSection_nVar(functname,tol,x, ...
        s,lowbound,intvl,ntrials);
    
    as(i+1) = output(1);
    fs(i+1) = output(2);
    for k = 1:nvar
        xs(i+1,k)=output(2+k);
        x(k)=output(2+k);
    end
    x
    grad= (gradfunction(functname,x)) % steepest descent
    
    convg(i+1)=grad*grad';
    % print convergence value
    fprintf('gradient length squared:  '),disp(convg(i+1));
    fprintf('objective function value:  '),disp(fs(i+1));
    %***********
    % draw lines
    %************
    
    if (nvar == 2)
        line([xs(i,1) xs(i+1,1)],[xs(i,2) xs(i+1,2)],'LineWidth',2, ...
            'Color',Lc)
        itr = int2str(i);
        x1loc = 0.5*(xs(i,1)+xs(i+1,1));
        x2loc = 0.5*(xs(i,2)+xs(i+1,2));
        %text(x1loc,x2loc,itr); 
        % writes iteration number on the line
        if strcmp(Lc,'r') 
            Lc = 'k';
        else
            Lc = 'r';
        end
        
        pause(1)  
        %***********************
        % finished drawing lines
        %***********************
    end
    
    if(convg(i+1)<= e3) break; end; % convergence criteria  
    % update the metric here
    
    delx = (x - xs(i,:))';
    gradold = gradfunction(functname,xs(i,:));
    Y = (grad -gradold)'; % column vector
    B = (Y*Y')/(Y'*delx);
    C = gradold'*gradold/(gradold*s');
    A = A + B + C;
    
    
    %***************************************
    %  complete the other stopping criteria
    %****************************************
end
len=length(as);
%for kk = 1:nvar
designvar=xs(length(as),:);
fprintf('The problem:  '),disp(functname)
fprintf('\nNo. of iterations:  '),disp(len)
fprintf('\nThe design vector,function value \nduring the iterations\n')
disp([xs fs']);
ReturnValue = [designvar fs(len)];