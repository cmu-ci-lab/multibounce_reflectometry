function [win,wout] = hd_to_io(h,d)

[phi_h,theta_h,~]=cart2sph(h(1),h(2),h(3));
theta_h=-theta_h+pi/2;

[phi_d,theta_d,~]=cart2sph(d(1),d(2),d(3));
theta_d=-theta_d+pi/2;

win=rotz(phi_h*180/pi)*roty(theta_h*180/pi)*d;
wout=2*dot(win,h)*h-win;

end

