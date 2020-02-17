function [PSNR] = GetPSNR( oI,I )
%UNTITLED5 Summary of this function goes here
%   Calculate the average SSIM score of the two hyperspectral images
% oI: the original image
% I: the recovery image

[m,n,p] = size(oI);
[m1,n1,p1] = size(I);
if m1 ~= m || n1~=n || p ~= p1
    error('Two image with different size!')
    PSNR = -Inf;
    return;
end
MSE = sum((oI(:) - I(:)).^2) * 1.0 / (m * n * p);
PSNR = 20 * log10(1) - 10 * log10(MSE);
end