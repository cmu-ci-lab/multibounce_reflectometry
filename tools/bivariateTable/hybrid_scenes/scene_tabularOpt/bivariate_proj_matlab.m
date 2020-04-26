function [bivariateTable_output] = bivariate_proj_matlab()

load('/home/kfirs/mitsuba-diff/tbsdf.mat');
bivariateTable = tbsdf;

run('/home/kfirs/cvx-a64/cvx/cvx_setup.m');

load('/home/kfirs/ref.mat');
load('/home/kfirs/BRDFtable.mat');
ref = squeeze(BRDFtable(90,:,:));

%load('/home/kfirs/rangeL.mat')
rangeL = [5,22,85];
rangeConvex = rangeL(rangeL > 40);

tbsdf = double(tbsdf);

observation_map = ones(90,90);
% observation_map = abs(double(tabulargrads));
observation_map(:,rangeL) = observation_map(:,rangeL) + 1e4; imagesc(log10(observation_map)); colorbar
W = diag(observation_map(:));

currX = tbsdf;
currX(:,rangeConvex) = ref(:,rangeConvex);
currX(currX < 1e-10) = 1e-10;

lambda = 1e3;
% quadprogW
cvx_begin
variable x(90,90)
y = x';
minimize sum( (W.^(1/2)*(x(:)-currX(:))).^2) + lambda * (1e-2*sum( (W_thetah*L*x(:)).^2) + sum( (W_thetad*L*y(:)).^2));
subject to
x >= 0
cvx_end

bivariateTable_output = x;

return;

end

