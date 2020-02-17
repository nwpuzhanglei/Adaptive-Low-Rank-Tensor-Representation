function [rX] = fun_LRTDMG(Y, mask, tau0, X)
% function of LRTDMG

opts.nBurnin = 40000;%200
opts.nCollect = 1000;%200
opts.D = 3;
opts.R = 20;%30
opts.time = false;
opts.display = false;
P = cp_als(tensor(Y),opts.R);
%opts.lambda = P.lambda;
opts.U = P.U;
rX = LRTDCF(Y, mask, tau0, opts, tensor(X));
rX = double(rX);
% rX = max(rX,0);
% rX = min(rX,1);
% rX(mask) = Y(mask);
end