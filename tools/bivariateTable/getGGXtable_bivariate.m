function [GGX] = getGGXtable_bivariate(alpha,eta)

theta=(0:89)/90 * pi/2;
theta_sqrt=(0:89).^2/90 * pi/180;

D=@(theta,alpha) 1/pi * alpha.^2 ./ (cos(theta).^2*(alpha^2-1)+1).^2;
G1= @(costheta,alpha)(costheta + sqrt(alpha^2+(1-alpha^2).*costheta.^2)).^(-1);

phi_h=0;
theta_h_idx=0;
phi=pi/2;
GGX=zeros(1,90,90);
for theta_h=theta_sqrt
    theta_h_idx=theta_h_idx+1;
    theta_d_idx=0;
    for theta_d=theta
        theta_d_idx=theta_d_idx+1;
        phi_d_idx=0;
        for phi_d=phi
            phi_d_idx=phi_d_idx+1;
            
            h=[sin(theta_h)*cos(phi_h);sin(theta_h)*sin(phi_h);cos(theta_h)];
            d=[sin(theta_d)*cos(phi_d);sin(theta_d)*sin(phi_d);cos(theta_d)];
            [win,wout] = hd_to_io(h,d);
            
            GGX(phi_d_idx,theta_h_idx,theta_d_idx) = F(win,h,eta) * D(theta_h,alpha) * G1(win(3),alpha)*G1(wout(3),alpha);
            
        end
    end
end

GGX = squeeze(GGX);
% GGX=repmat(GGX,[180,1,1]);
end
