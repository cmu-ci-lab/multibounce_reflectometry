function [A,C] = getDictionaryMat(dictMat,W)
% This function gets dictMat [Rx2] when R is the number of atoms.
% for each 1<=r<=R:
% dict(r,1) - alpha
% dict(r,2) - eta



N = 90 * 90; % elements of bivariate table
R = size(dictMat,1);
A = zeros(N, R);
for r=1:R
    currAtom = getGGXtable_bivariate(dictMat(r,1),dictMat(r,2));
    A(:,r) = currAtom(:);
end

W = diag(W(:));

W_sqrt = sqrt(W);

A = W_sqrt*A;

% C = A * (A'*diag(W)*A)^(-1) * A' * diag(W) - eye(N);

C = A * pinv(A) - eye(N);