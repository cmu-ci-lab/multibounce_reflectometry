
theta=(0:89)/90 * pi/2;
phi=(0:179)/180 * pi/2;
theta_sqrt=(0:89).^2/90 * pi/180;

phi_h=0;
theta_h_idx=0;
val=zeros(length(phi),90,90);
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
            
            val(phi_d_idx,theta_h_idx,theta_d_idx) = win(3);
        end
    end
end


weightCos_theta_in = squeeze(max(val , [] , 1));
weightCos_theta_in = max(weightCos_theta_in,0);
imshow(weightCos_theta_in , [])
save('weightCos_theta_in','weightCos_theta_in');

% for n=1:180
%     imshow(squeeze(val(n,:,:)),[])
%     title(num2str(n));
%     pause(0.4);
% end

