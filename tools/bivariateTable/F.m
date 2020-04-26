function val = F(v,m,eta)
c=abs(dot(v,m));

% theta=linspace(0,pi/2,100);
% c=cos(theta);
g=sqrt(eta.^2-1+c.^2);

gmc=g-c;
gpc=g+c;

val = 0.5 .* (gmc./(gpc+eps)).^2 .* hypot(1,(c.*gpc-1)./(c.*gmc+1+eps)).^2;

% g=sqrt(eta.^2-1+c.^2);
% cmg=c-g;
% cpg=c+g;
% diffTerm = - (cmg^2*((2*c*eta*abs(c*cpg - 1)^2*sign(1 - c*cmg))/(abs(1 - c*cmg)^3*g) - (2*c*eta*abs(c*cpg - 1)*sign(c*cpg - 1))/(abs(1 - c*cmg)^2*g)))/(2*cpg^2) - (eta*cmg^2*(abs(c*cpg - 1)^2/abs(1 - c*cmg)^2 + 1))/(cpg^3*g) - (eta*cmg*(abs(c*cpg - 1)^2/abs(1 - c*cmg)^2 + 1))/(cpg^2*g);

end

