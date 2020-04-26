
load('WeightMat');
W = table;
W = W / sum(W(:));
W = W(:);

load('W2');
W = W .* W2(:);

load('reduced_dict_45.mat');
dictMat = [];
counter = 0;
for n=1:size(BRDF_dict,1)
    if BRDF_dict(n,2) ~= -1
        counter = counter + 1;
        dictMat(counter,1) =  BRDF_dict(n,1);
        dictMat(counter,2) =  BRDF_dict(n,2);
    end
end

[A,C] = getDictionaryMat(dictMat,W);

CW = C * diag(W(:))^(1/2);
M = (CW'*CW);

save('M','M');

% the regularization is X'*M*X
