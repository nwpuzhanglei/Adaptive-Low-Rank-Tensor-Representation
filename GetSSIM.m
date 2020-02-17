function [ssim1] = GetSSIM( oI,I)
%UNTITLED5 Summary of this function goes here
%   Calculate the average SSIM score of the two hyperspectral images
% oI: the original image
% I: the recovery image

[m,n,p] = size(oI);
[m1,n1,p1] = size(I);
if m1 ~= m || n1~=n || p ~= p1
    error('Two image with different size!')
    ssim1 = -Inf;
    return;
end
score = 0;
for i = 1 : p
    img1 = oI(:,:,i) .* 255;
    img2 = I(:,:,i) .* 255;
    score = score + fun_ssim(img1, img2);
end
ssim1 = score * 1.0 / p;
end

