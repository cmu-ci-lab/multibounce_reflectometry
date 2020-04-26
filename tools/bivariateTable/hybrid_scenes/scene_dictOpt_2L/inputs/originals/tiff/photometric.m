% CS283 Fall 2015
% Student: Ioannis Gkioulekas
% Homework assignment 6, question 4
%
% Photometric stereo.

%% Part a

% load data
load ./sources.mat
%load ./data/PhotometricStereo/sources.mat
S(:,3) = -S(:,3); 
S(:,2) = -S(:,2);
%S = -S

% find number of images
numImgs = size(S, 1);

% load first image and make it grayscale and double
temp = im2double(imread('./0000.tif'));
%temp = im2double(rgb2gray(imread('./data/PhotometricStereo/female_01.tif')));

% find number of pixels and image dimensions
P = numel(temp);
[M N] = size(temp);

% stack images in large matrix, one image per row (so, one column of this
% matrix corresponds to the 7 measurements of one pixel) 
I = zeros(numImgs, P);

% copy first image
I(1, :) = temp(:);

% copy other images, after converting to grayscale and double
for iter = 1:numImgs-1,
    temp = im2double(imread(...
        strcat('./000', num2str(iter), '.tif')));
	I(iter+1, :) = temp(:);
end;%strcat('./data/PhotometricStereo/female_0', num2str(iter), '.tif'))))
		%strcat('./data/PSTest0/ostereo-lt-', num2str(iter), '.tif'))));

% calculate vector b by solving linear system in least-squares sense
%b = S \ I;

b = zeros(3, P);
for i = (1:P)
    tI = I(:,i);
    tS = S;
    [k, idxs] = sort(tI);
    b(:,i) = tS(idxs(2:numImgs), :) \ tI(idxs(2:numImgs));
end

% calculate albedo
p = sqrt(sum(b .^ 2));
p = reshape(p, M, N);

% calculate components of normal vector
n = reshape(transpose(b), size(temp, 1), size(temp, 2), 3) ./ ...
	repmat(p, [1 1 3]);

skip=2
% plot vector field of normal vectors
figure; quiver(n(1:skip:end, 1:skip:end, 1), n(1:skip:end, 1:skip:end, 2));
axis image; axis ij; axis off;

% plot albedo
figure; imshow(p);

[h,w]=size(n(:,:,1));

%% Part X
% complain if P or Q are too big
if (h>512) | (w>512)
  error('Input array too big.  Choose a smaller window.');
end

% pad the input array to 512x512
nrows=2^9; ncols=2^9;

% get surface slopes from normals; ignore points where normal is [0 0 0]
x_sample=1;
y_sample=1;
zx=-x_sample*(sum(n,3)~=0).*n(:,:,1)./(n(:,:,3)+(n(:,:,3)==0));
zy=-y_sample*(sum(n,3)~=0).*n(:,:,2)./(n(:,:,3)+(n(:,:,3)==0));   

zx(isnan(zx)) = 0;
zy(isnan(zy)) = 0;


Z = DepthFromGradient(zx, zy, struct('periodic', 0));
%% Part c

% the two different normal vectors
s1 = [0.58 -0.58 -0.58];
s2 = [-0.58 -0.58 -0.58];

% calculate inner products to find radiance under two different
% illumination conditions
Is1 = max(p .* (n(:,:,1) * s1(1) +...
		n(:,:,2) * s1(2) + n(:,:,3) * s1(3)), 0);
Is2 = max(p .* (n(:,:,1) * s2(1) +...
		n(:,:,2) * s2(2) + n(:,:,3) * s2(3)), 0);

% plot results
figure; imshow(Is1);
figure; imshow(Is2);

csvwrite('normal_field_1.csv', flipud(n(:, :, 1)))
csvwrite('normal_field_2.csv', flipud(n(:, :, 2)))
csvwrite('normal_field_3.csv', flipud(n(:, :, 3)))
%% Part d

% integrate normal vectors
% Z = integrate_frankot(n);

% plot surface
figure; s = surf(-Z);
axis image; axis off;
set(s, 'facecolor', [.5 .5 .5], 'edgecolor', 'none');
l = camlight;
rotate3d on

