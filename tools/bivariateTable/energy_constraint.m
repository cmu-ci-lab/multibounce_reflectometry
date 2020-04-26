
% run over theta_in in order to get energy conservation
theta_in_range = 0:1:90;
counterThetaIn = 0;

% define parameters resolution
N = 500;
theta = linspace(0,pi/2,N);
phi = linspace(0,2*pi,4*N);
MERL_THETAD_RES=90;
MERL_THETAH_RES=90;

energyTable = zeros(MERL_THETAH_RES , MERL_THETAD_RES , length(theta_in_range));

% loop over theta_in
for theta_in=theta_in_range
    counterThetaIn = counterThetaIn + 1
    
    phi_in = 0;
    win = [sind(theta_in)*cosd(phi_in);sind(theta_in)*sind(phi_in);cosd(theta_in)];
    
    table = zeros(MERL_THETAH_RES,MERL_THETAD_RES);
    s = 0;
    % for each theta_in, the integral over hemisphere is less than 1.
    % get the mapping for hd space.
    for theta_out=theta
        for phi_out=phi
            wout = [sin(theta_out)*cos(phi_out);sin(theta_out)*sin(phi_out);cos(theta_out)];
            [h,d,theta_h,phi_h,theta_d,phi_d] = io_to_hd(win,wout);
            
            % get theta_d idx - linear mapping
            theta_d_idx=theta_d/(pi/2) * MERL_THETAD_RES + 0.5;
            theta_d_idx=round(theta_d_idx);
            
            % get theta_h idx - non-linear mapping
            theta_h_idx=sqrt(theta_h/(pi/2) * MERL_THETAH_RES^2);
            theta_h_idx=round(theta_h_idx);
            theta_h_idx=max(0,theta_h_idx); theta_h_idx=min(theta_h_idx,89);
            theta_h_idx=theta_h_idx+1;
            
            table(theta_h_idx,theta_d_idx) = table(theta_h_idx,theta_d_idx) + (2*pi)/(4*N) * (pi/2)/N * cos(theta_out) * sin(theta_out);
            
            s = s + (2*pi)/(4*N) * (pi/2)/N * cos(theta_out) * sin(theta_out);
        end
    end
    
    energyTable(:,:,counterThetaIn) = table;
end

%% create parameter to save
hd_weightMat = reshape(energyTable , [MERL_THETAD_RES*MERL_THETAH_RES , size(energyTable,3)]);
hd_weightMat = hd_weightMat';

save('hd_weightMat', 'hd_weightMat');


