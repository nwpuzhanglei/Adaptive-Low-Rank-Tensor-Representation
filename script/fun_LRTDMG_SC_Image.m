function [rX] = fun_LRTDMG_SC_Image(Y, mask, tau0, X)
% function of LRTDMG

Y = permute(Y,[3,1,2]);
X = permute(X,[3,1,2]);
mask = permute(mask, [3,1,2]);

opts.nBurnin = 300;% samples for burin
opts.nCollect = 300;% samples for compute the mean, in general, more samples will lead more stable result
opts.D = 3; % number of Gassuian component in the Gaussian mixture model
opts.R = 30; % R parameter in Eq.(5) of the main manuscript
opts.time = false;
opts.display = true; % true: 
opts.SC = true; % introducing the spatial constraint

InitiX = Y;

P = cp_als(tensor(InitiX),opts.R,'init','nvecs');
%opts.lambda = P.lambda;
opts.U = P.U;
rX = LRTDCF_SC(Y, mask, tau0, opts, tensor(X));
rX = double(rX);
rX = max(rX,0);
rX = min(rX,1);
rX(mask) = Y(mask);

rX = permute(rX,[2,3,1]);

end