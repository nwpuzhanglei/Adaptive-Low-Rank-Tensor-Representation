function [RSE] = GetRSE( oI,I )
%UNTITLED5 Summary of this function goes here
%   Calculate the average SSIM score of the two hyperspectral images
% oI: the original image
% I: the recovery image

[m,n,p] = size(oI);
[m1,n1,p1] = size(I);
if m1 ~= m || n1~=n || p ~= p1
    error('Two image with different size!')
    RSE = -Inf;
    return;
end
diff = oI - I;
RSE = norm(diff(:)) ./ norm(oI(:));
end