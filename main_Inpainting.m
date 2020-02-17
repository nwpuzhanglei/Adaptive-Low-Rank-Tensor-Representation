% main function for image inpainting

addpath(genpath('./'));

tau0 = 800; % predefined precision of nois corruption in sampling

rand('seed',0);
randn('seed',0);

%1. Read image
imName = './data/facade.bmp';
X = double(imread(imName)) ./ 255.0;
Nway = size(X); % image size
Nel = numel(X); % element number in image


%2. Generate mask
SR = 0.2; % sampling rate, the missing rate is (1 - SR), 80% pixels are randomly missing.
srNum = round(SR * Nel);
index = randsample(Nel, srNum);
mask = zeros(Nel,1);
mask(index) = 1;
mask = logical(reshape(mask,Nway));

%3. Sample image with or without noise
Y = X;
%Y = X + sqrt(1.0 / tau0) * randn(size(X));
Y = Y .* mask;

rX = fun_LRTDMG_SC_Image(Y, mask, tau0, X);% proposed method
psnr = GetPSNR(X,rX);
ssim = GetSSIM(X,rX);
rse = GetRSE(X,rX);
fprintf(['rse = %4.2e, psnr = %.4f, ssim = %.4f; \n'], rse, psnr, ssim);
