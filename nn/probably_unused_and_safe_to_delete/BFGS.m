function [xn]=BFGS(fg,x0,tol,B0)
%
%
% S. Ulbrich, May 2002
%
% This code comes with no guarantee or warranty of any kind.
%
% function [xn]=BFGS(x0,fg,tol,B0)
%
% BFGS-method with Powell-Wolfe stepsize rule. 
%
% Input:  x0      starting point
%         fg      name of a matlab-function [f,g]=fg(x)
%                 that returns value and gradient
%                 of the objective function depending on the
%                 number of the given ouput arguments
%         tol     stopping tolerance: the algorithm stops
%                 if ||g(x)||<=tol*max(1,||g(x0)||)
%         B0      initial BFGS-matrix (optional)
%                 if not given B0=I is used
%
% Output: xn      result after termination
%

% constants 0<del<theta<1, del<1/2 for Wolfe condition
del=0.001;
theta=0.6;
% constant 0<al<1 for sufficient decrease condition
al=0.001;

xj=x0;
[f,g]=feval(fg,xj);
nmg0=norm(g);
nmg=nmg0;
it=0;
if nargin<4
 B=eye(size(g,1));
else
 B=B0;
end
% main loop
while (norm(g)>tol*max(1,nmg0))
 it=it+1;
 sig=1;
% compute BFGS-step
 s=B*g;
 step='BFGS';

% check if BFGS-step provides sufficient decrease; else take gradient
 stg=s'*g;
 if stg<min(al,nmg)*nmg*norm(s)
  s=g;
  stg=s'*g;
  step='Grad';
 end
% choose sig by Powell-Wolfe stepsize rule
 sig=wolfe(xj,s,stg,fg,f,del,theta,1.0);
 xn=xj-sig*s;
 fprintf(1,'it=%3.d   f=%e   ||g||=%e   sig=%5.3f   step=%s\n',it,f,norm(g),sig,step);
 [fn,gn]=feval(fg,xn);
% update BFGS-matrix
 d=g-gn;
 p=xj-xn;
 dtp=d'*p;
 if dtp>=1e-8*norm(d)*norm(p)
  Bd=B*d;
  B=B+(dtp+d'*Bd)/(dtp*dtp)*p*p'-(1/dtp)*(p*Bd'+Bd*p');
 end
 xj=xn;
 g=gn;
 f=fn;
 nmg=norm(g);
end
it=it+1;
fprintf(1,'it=%3.d   f=%e   ||g||=%e\n\n',it,f,norm(g));
fprintf(1,'Successful termination with ||g||<%e*max(1,||g0||):\n',tol);
