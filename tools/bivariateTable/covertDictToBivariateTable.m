function bivariateTable = covertDictToBivariateTable(dictMat)
% This function gets dictionary of GGX (with alpha and eta) and their
% weight for each atom and return the bivariate table of them.
% Input: dictMat Rx3, R is the length of the dictionary.
% for each 1<=r<=R:
% dictMat(r,1) - alpha
% dictMat(r,2) - eta
% dictMat(r,3) - weight

R = size(dictMat,1);
bivariateTable = zeros(90,90);
for r=1:R
    bivariateTable = bivariateTable + dictMat(r,3) * getGGXtable_bivariate(dictMat(r,1),dictMat(r,2)) ;
end

end

