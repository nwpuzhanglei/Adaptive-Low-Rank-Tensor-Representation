function [ rL ] = LRTDCF_SC(Y, Mask, tau0, opts, L0)
% FUN: Low Rank Decompositionn with spatially constrained factors
%
%% INPUT:
%
%   Y,                  incompleted noisy tensor (observation)
%   Mask,               indicator tensor
%   tau0,               noise precision
%   mask,               mask for incomplete tensor Y
%   opts,               initialized parameters for Gibbs sampling
%       'nBurnin',      scalar, number of burnin iterations, default is 200
%       'nCollect',     scalar, number of collected samples, default is 100
%       'nInterval'     scalar, samples are collected very nInterval
%                       interations, detault is 1
%       'R'             scalar, initialized rank, default is 100
%       'lambda'        vector, [R, 1], weight vector in CP decomposition,
%       'U'             cell, factor matrices
%       'gamma'         vector, [R,1], variance vector in sparsity prior
%       'kappa'         vector, [R,1], hyperparameter vector in sparsity prior
%       'a0'            scalar, hyperparameters for M,Me,T,Te
%       'b0'            scalar, hyperparameters for M,Me,T,Te
%       'D'             vector, [K,1], cluster number in all factors
%       'PI'            cell, weights for mixture components in all factors
%       'M'             vector, means for K factors
%       'T'             vector, precisions for K factors
%       'Me'            cell, means for mixture compomnents in residual component
%       'Te'            cell, precisions for mixture compomnents in residual component
%       'Z'             matrix, indicators for mixture components in residual component
%       'mu0'           scalar, hyperparameters for M,T
%       'beta0'         scalar, hyperparameters for M,T
%       'eta0'          scalar, hyperparameters for spatial constraint
%       'A0'            cell, hyperparameters for PI
%       'saveSample'    logical, default is false (samples are not saved)
%       'dispaly'       logical, default is false (display infor)
%       'SC'            logical, default is false (spatial variational constraint)
%
%% OUTPUT:
%
%       rX,             recovered tensor
%
%


%% Parameter Settings
Nway = size(Y);
K = length(Nway);
Nc = prod(Nway);% number of component in desired tensor
E = tensor(zeros(Nway));
%stdv = std(Y(:));
stdv = 1;
%stdv = max(abs(Y(:)));
Y = tensor(Y ./ stdv);
[nzIndex, nzVal0] = find(Y); % subscripts and values for nonzero elements in Y
nzIndexCell = mat2cell(nzIndex', ones(size(nzIndex,2),1), size(nzIndex,1));% subscripts and values for nonzero elements in Y: cell
nzNum = length(nzVal0);

O = reshape(Mask, Nc,1); % indicator tensor;
nzIndexO = tt_sub2ind(Nway,nzIndex);

pIndex = cell(K,1);% partical index for each subscripts of nonzero elements in Y
Ws = cell(K,1); % weight for spatial constraint
for k = 1 : K
    pIndex{k} = cell(Nway(k),1);
    for i = 1 : Nway(k)
        pIndex{k}{i} = find(nzIndex(:,k) == i);
    end
    Ws{k} = GetWeight(Nway(k), nzNum / Nc);
end

if isfield(opts, 'nBurnin')
    nBurnin = opts.nBurnin;
else
    nBurnin = 200;
end

if isfield(opts, 'nCollect')
    nCollect = opts.nCollect;
else
    nCollect = 100;
end

if isfield(opts, 'nInterval')
    nInterval = opts.nInterval;
else
    nInterval = 1;
end

if isfield(opts, 'R')
    R = opts.R;
else
    R = 20;
end

if isfield(opts, 'lambda')
    lambda = opts.lambda;
else
    lambda = ones(R,1);
end

if isfield(opts, 'U')
    U = opts.U;
else
    U = cell(K,1);
    for k = 1 : K
        U{k} = ones(Nway(k),R);
    end
end

if isfield(opts, 'gamma')
    gamma = opts.gamma;
else
    gamma = ones(R,1);
end

if isfield(opts, 'kappa')
    kappa = opts.kappa;
else
    kappa = ones(R,1);
end

if isfield(opts, 'a0')
    a0 = opts.a0;
else
    a0 = 1e-6;
end

if isfield(opts, 'b0')
    b0 = opts.b0;
else
    b0 = 1e-6;
end

if isfield(opts, 'D')
    D = opts.D;
else
    D = 6;
end

if isfield(opts, 'PI')
    PI = opts.PI;
else
    temp = ones(D,1);
    PI = temp ./ sum(temp);
end

if isfield(opts, 'M')
    M = opts.M;
else
    M = zeros(K,1);
end

if isfield(opts, 'T')
    T = opts.T;
else
    T = ones(K,1);
end

if isfield(opts, 'Me')
    Me = opts.Me;
else
    Me = zeros(D,1);
end

if isfield(opts, 'Te')
    Te = opts.Te;
else
    Te = ones(D,1);
end

if isfield(opts, 'Z')
    Z = opts.Z;
else
    temp = ones(1,D);
    temp = temp ./ sum(temp);
    updf = repmat(temp, Nc, 1);
    Z = mnrnd(1,updf);
end

if isfield(opts, 'mu0')
    mu0 = opts.mu0;
else
    mu0 = 0;
end

if isfield(opts, 'beta0')
    beta0 = opts.beta0;
else
    beta0 = 1e-6;
end

if isfield(opts, 'eta0')
    eta0 = opts.eta0;
else
    eta0 = 1e3;
end

if isfield(opts, 'A0')
    A0 = opts.A0;
else
    A0 = ones(D,1) .* 1e-6;
end

if isfield(opts, 'saveSample')
    saveSample = opts.saveSample;
else
    saveSample = false;
end

if ~saveSample
    rlambda = zeros(R,1);
    rU = cell(K,1);
    rM = zeros(K,1);
    rT = zeros(K,1);
    rMe = zeros(D,1);
    rTe = zeros(D,1);
    rPI = zeros(D,1);
    rZ = zeros(Nc,D);
    rE = tensor(zeros(Nway));
    
    mlambda = zeros(R,1);
    mU = cell(K,1);
    mE = tensor(zeros(Nway));
    
    for k = 1 : K
        rU{k} = zeros(Nway(k),R);
        mU{k} = zeros(Nway(k),R);
    end
    AError = zeros(nCollect,1);
    SError = zeros(nCollect,1);
end


%% Sampling
snum = 0; % number of samples
if opts.display
    fprintf('samples:     ');
end
nzE = E(nzIndex);
for iter = 1 : (nBurnin + nCollect)
    
    if opts.display
        fprintf('\b\b\b\b\b%5i',iter);
    end
    
    % X = Y - E
    nzVal = nzVal0 - nzE;
    
    % 1. sample lambda
    if opts.time
        tic
        lambda = SampleLambda(nzIndexCell, nzVal, tau0, lambda, U, gamma, R);
        t1 = toc;
        fprintf('\nlambda : %.4fs\n', t1);
    else
        lambda = SampleLambda(nzIndexCell, nzVal, tau0, lambda, U, gamma, R);
    end
    
    % 2. sample gamma
    if opts.time
        tic
        gamma = SampleGamma(lambda, gamma, kappa, R);
        t2 = toc;
        fprintf('gamma : %.4fs\n', t2);
    else
        gamma = SampleGamma(lambda, gamma, kappa, R);
    end
    
    % 3. sample kappa
    if opts.time
        tic
        kappa = SampleKappa(gamma);
        t3 = toc;
        fprintf('kappa : %.4fs\n', t3);
    else
        kappa = SampleKappa(gamma);
    end
    
    % 4. sample U
    if opts.time
        tic
        U = SampleU(nzIndex, nzIndexCell, pIndex, nzVal, tau0, lambda, U, M, T, R, K, Nway, opts.SC, Ws, eta0);
        t4 = toc;
        fprintf('U : %.4fs\n', t4);
    else
        U = SampleU(nzIndex, nzIndexCell, pIndex, nzVal, tau0, lambda, U, M, T, R, K, Nway, opts.SC, Ws, eta0);
    end
    
    % 5. sample M
    if opts.time
        tic
        M = SampleM(U, M, T, mu0, beta0, K);
        t5 = toc;
        fprintf('M : %.4fs\n', t5);
    else
        M = SampleM(U, M, T, mu0, beta0, K);
    end
    
    % 6. sample T
    if opts.time
        tic
        T = SampleT(U, T, M, mu0, beta0, a0, b0, K);
        t6 = toc;
        fprintf('T : %.4fs\n', t6);
    else
        T = SampleT(U, T, M, mu0, beta0, a0, b0, K);
    end
    
    % 7. sample E
    if opts.time
        tic
        [E, nzE] = SampleE(Y, lambda, U, O, Z, tau0, Te, Me, Nc, Nway, nzIndexO);
        t7 = toc;
        fprintf('E : %.4fs\n', t7);
    else
        [E, nzE] = SampleE(Y, lambda, U, O, Z, tau0, Te, Me, Nc, Nway, nzIndexO);
    end
    
    % 8. sample Me
    if opts.time
        tic
        Me = SampleMe(nzE, Te, Z(nzIndexO,:), mu0, beta0, D);
        t8 = toc;
        fprintf('Me : %.4fs\n', t8);
    else
        Me = SampleMe(nzE, Te, Z(nzIndexO,:), mu0, beta0, D);
    end
    
    % 9. sample Te
    if opts.time
        tic
        Te = SampleTe(nzE, Me, Z(nzIndexO,:), mu0, beta0, a0, b0, D, nzNum);
        t9 = toc;
        fprintf('Te : %.4fs\n', t9);
    else
        Te = SampleTe(nzE, Me, Z(nzIndexO,:), mu0, beta0, a0, b0, D, nzNum);
    end
    
    if D > 1
        
        % 10. sample Z
        if opts.time
            tic
            Z = SampleZ(E, Me, Te, PI, D, Nc);
            t10 = toc;
            fprintf('Z : %.4fs\n', t10);
        else
            Z = SampleZ(E, Me, Te, PI, D, Nc);
        end
        
        % 11. sample PI
        if opts.time
            tic
            PI = SamplePI(Z(nzIndexO,:), A0);
            t11 = toc;
            fprintf('PI : %.4fs\n', t11);
        else
            PI = SamplePI(Z(nzIndexO,:), A0);
        end
        
    end
        
    if iter >  nBurnin && (mod(iter - nBurnin, nInterval) == 1 || nInterval == 1)
        snum = snum + 1;
        if saveSample
            samples(snum).lambda = lambda;
            samples(snum).U = U;
            samples(snum).M = M;
            samples(snum).T = T;
            samples(snum).Z = Z;
            samples(snum).PI = PI;
        else
            rlambda = rlambda + lambda;
            mlambda = rlambda ./ snum;
            rM = rM + M;
            rT = rT + T;
            rMe = rMe + Me;
            rTe = rTe + Te;
            rPI = rPI + PI;
            rZ = rZ + Z;
            rE = plus(rE,E);
            for k = 1 : K
                rU{k} = rU{k} + U{k};
                mU{k} = rU{k} ./ snum;
            end
            mE = mtimes(rE,1.0 / snum);
            
            mL = full(ktensor(mlambda,mU));
            mL = plus(mL,mE);
            mL = mtimes(mL,stdv);
            
            sL = full(ktensor(lambda,U));
            sL = plus(sL,E);
            sL = mtimes(sL,stdv);
            
            AError(snum) = norm(minus(mL,L0)) / norm(L0);
            SError(snum) = norm(minus(sL,L0)) / norm(L0);
            
        end
    end
end
if opts.display
    fprintf('\n');
end

if opts.display
    figure,plot(1:nCollect,[AError,SError]);
    legend('Average Err', 'Sample Err');
end

%% Estimation
if saveSample
    [rX, rlambda, rU, rM, rT, rPI] = GibbsEst(samples, K, R, Nway, D, snum);
else
    rlambda = rlambda .* 1.0 ./ snum;
    rM = rM .* 1.0 ./ snum;
    rT = rT .* 1.0 ./ snum;
    rMe = rMe .* 1.0 ./ snum;
    rTe = rTe .* 1.0 ./ snum;
    rPI = rPI .* 1.0 ./ snum;
    rZ = rZ .* 1.0 ./ snum;
    rE = mtimes(rE,1.0 / snum);
    for k = 1 : K
        rU{k} = rU{k} .* 1.0 ./ snum;
    end
    rL = full(ktensor(rlambda,rU));
    rL = plus(rL,rE);
    rL = mtimes(rL,stdv);
    Error = norm(minus(rL,L0)) / norm(L0);
    if opts.display
        fprintf('final error %4.2e \n',Error);
    end
end

end

function lambda = SampleLambda(nzIndexCell, nzVal, tau0, lambda, U, gamma, R)
% lambda is sampled from a Gaussian distribution
% Note: entries in lambda are not independent. Thus, we have to sample them
% together from their joint distribution or sample each entries
% independently conditioned on other entries. Since the joint
% distributution is hard to obtain, we adopt the second way.
fun = @(X,index) X(index,:);
Uks = cellfun(fun, U,  nzIndexCell, 'UniformOutput', false);
Uks = cat(3,Uks{:});
birs = prod(Uks,3);
bir2 = sum(birs.^2,1);
for r = 1 : R
    temp = bsxfun(@times, birs, lambda');
    
    bir = birs(:,r);
    tsum = sum(temp,2);
    yir = nzVal - tsum + temp(:,r);
    
    taus = 1.0 ./ (gamma(r) + eps) + tau0 .* bir2(r); % precisions in the Gaussian distirbution for each element in lambda each
    mus = (tau0 ./ (taus + eps)) .* sum(bir .* yir,1); % means in the Gaussian distirbution for each element in lambda
    lambda(r) = normrnd(mus, sqrt(1.0 ./ (taus + eps)));
end
end

function gamma = SampleGamma(lambda, gamma, kappa, R)
% gamma is sampled from a generalized inverse Gaussian distribution

for r = 1 : R
    gamma(r) = gigrnd(0.5, kappa(r), (lambda(r))^2, 1);
end

end

function kappa = SampleKappa(gamma)
% kappa is sampled from a Gamma distribution

%kappa = gamrnd((1 + a0) .* ones(R,1), (2.0 * b0) ./ (gamma .* b0 + 2));
kappa = gamrnd(2, 2.0 ./ (gamma + eps));

end

function U = SampleU(nzIndex, nzIndexCell, pIndex, nzVal, tau0, lambda, U, M, T, R, K, Nway, SC, Ws, eta)
% U is sampled from a Gaussian distribution
% Note: entries in U are not independent. Thus, we have to sample them
% together from their joint distribution or sample each entries
% independently conditioned on other entries. Since the joint
% distributution is hard to obtain, we adopt the second way. Since the
% entries in each column of U{k} are independent, we sample each column of U{k} at once.
fun = @(X,index) X(index,:);
for k = 1 : K
    yirks = cellfun(fun, U,  nzIndexCell, 'UniformOutput', false);
    yirks = cat(3,yirks{:});
    yirks = prod(yirks,3);
    temp = bsxfun(@times, yirks, lambda');
    cirks = yirks ./ U{k}(nzIndex(:,k),:);
    cirks = bsxfun(@times, cirks, lambda');
    
    for r = 1 : R
        Mus = ones(Nway(k),1) .* M(k); % means in the Gaussian distirbution for each element in U^(k)
        Taus = ones(Nway(k),1) .* T(k); % precision in the Gaussian distirbution for each element in U^(k)
        
        tsum = sum(temp,2);
        yirk = nzVal - tsum + temp(:,r);
        
        cirk = cirks(:,r);
        yirk = cirk .* yirk;
        cirk = cirk.^2; %
        
        if SC && k > 1
            W = Ws{k};
            Utemp = repmat((U{k}(:,r))',Nway(k),1);
            Btemp = ones(Nway(k),Nway(k)) .* eta;
            Mtemp = Utemp - diag(diag(Utemp) - M(k));
            Ttemp = Btemp - diag(diag(Btemp) - T(k));
            MTk = diag((Mtemp .* Ttemp) * W);
            Tk = diag(Ttemp * W);
        end
        
        for i = 1 : Nway(k)
            index = pIndex{k}{i};
            num = length(index);
            if num > 0
                if SC && k > 1
                    Taus(i) = tau0 .* sum(cirk(index),1);
                    Mus(i) = tau0 .* sum(yirk(index),1);
                else
                    Taus(i) = tau0 .* sum(cirk(index),1) + T(k);
                    Mus(i) = (tau0 .* sum(yirk(index),1) + T(k) * M(k)) ./ (Taus(i) + eps);
                end
            end
        end
        
        if SC && k > 1
            Taus = Taus + Tk;
            Mus = (Mus + MTk) ./ (Taus + eps);
        end
        
        temp(:,r) = temp(:,r) ./ U{k}(nzIndex(:,k),r);
        U{k}(:,r) = normrnd(Mus, sqrt(1.0 ./ (Taus + eps)));
        temp(:,r) = temp(:,r) .* U{k}(nzIndex(:,k),r);
    end
end

end

function M = SampleM(U, M, T, mu0, beta0, K)
% M is sampled from a Gaussian distribution

for k = 1 : K
    [nk,R] = size(U{k});
    sumUk = sum(sum(U{k}));
    taus = T(k) .* (beta0 + nk * R);
    mus = T(k) ./ (taus + eps) .* (sumUk + beta0 * mu0);
    M(k) = normrnd(mus, sqrt(1.0 ./ (taus + eps)));
end

end

function T = SampleT(U, T, M, mu0, beta0, a0, b0, K)
% T is sampled from a Gaussian distribution

for k = 1 : K
    [nk,R] = size(U{k});
    sumUk = sum(sum((U{k} - M(k)).^2));
    as = a0 + (1 + nk * R).* 0.5;
    bs = b0 + sumUk * 0.5 + beta0 * (M(k)- mu0)^2 * 0.5;
    T(k) = gamrnd(as, 1.0 ./ bs);
end

end

function nzE = SampleE2(lambda, U, Z, tau0, Te, Me, nzIndex, nzVal0)
% E is sampled from a Gaussian distribution

Num = length(nzVal0);
X = full(ktensor(lambda,U));
Xv = X(nzIndex);
Yx = (nzVal0 - Xv) .* tau0;
TZ = repmat(Te', Num, 1) .* Z;
TMZ = TZ .* repmat(Me', Num, 1);
taus = tau0 + sum(TZ,2);
mus = (Yx + sum(TMZ,2)) ./ (taus + eps);
nzE = normrnd(mus, sqrt(1.0 ./ (taus + eps)));

end

function [E, nzE] = SampleE(Y, lambda, U, O, Z, tau0, Te, Me, Nc, Nway, nzIndexO)
% E is sampled from a Gaussian distribution

X = full(ktensor(lambda,U));
Yx = double(minus(Y,X));
Yx = reshape(Yx, Nc, 1) .* O .* tau0;

TZ = bsxfun(@times, Z, Te');
TMZ = bsxfun(@times, TZ, Me');

taus = O * tau0 + sum(TZ,2);
mus = (Yx + sum(TMZ,2)) ./ (taus + eps);
Index = [1 : Nc]';
Index(nzIndexO) = [];
nzE = normrnd(mus(nzIndexO), sqrt(1.0 ./ (taus(nzIndexO) + eps)));
E(nzIndexO,:) = nzE;
pp = gamrnd(1,1);% separate (必须阻断一下，要不然random出来的数据过于相关)
E(Index,:) = normrnd(mus(Index), sqrt(1.0 ./ (taus(Index) + eps)));
E = tensor(reshape(E,Nway));
end

function Me = SampleMe(nzE, Te, Z, mu0, beta0, D)
% Me is sampled from a Gaussian distribution

Es = repmat(nzE,1,D);
sumZ = sum(Z,1) + beta0;
sumZe = sum(Z .* Es,1) + beta0 * mu0;

taus = Te .* sumZ';
mus = Te .* sumZe' ./ (taus + eps);

Me = normrnd(mus, sqrt(1.0 ./ (taus + eps)));

end

function Te = SampleTe(nzE, Me, Z, mu0, beta0, a0, b0, D, nzNum)
% Te is sampled from a Gaussian distribution

sumZ = sum(Z,1) + 1;
Es = (repmat(nzE,1,D) - repmat(Me', nzNum, 1)).^2;
sumZe = sum(Z .* Es,1);
tempMe = beta0 .* (Me - mu0).^2;

as = a0 + sumZ' .* 0.5;
bs = b0 + (sumZe' + tempMe) .* 0.5;

Te = gamrnd(as, 1.0 ./ bs);

end

function Z = SampleZ(E, Me, Te, PI, D, Nc)
% Z is sampled from a multinomial distribution

Es = reshape(double(E), Nc, 1);
updf = zeros(Nc,D);
for i = 1 : D
    updf(:,i) = normpdf(Es, Me(i), sqrt(1.0 / (Te(i) + eps))) .* PI(i);
end
updf = updf + eps;
updf = updf ./ repmat(sum(updf,2),1,D);
Z = double(mnrndC(updf'));
Z = Z';
%Z = mnrnd(1,updf);
end

function PI = SamplePI(Z, A0)
% PI is sampled from a Dirichlet distribution

sumZ = sum(Z,1);
alpha = sumZ' + A0;
PI = drchrnd(alpha);

end

function [W] = GetWeight(N, Ro)
% Get weight in the mixture prior for U
% Output:
%   W    weight matrix

W = ones(N,N);
for i = 1 : N
    for j = i + 1 : N
        W(j,i) = exp(- 2 * Ro * abs(i - j)^2);
        W(i,j) = W(j,i);
    end
end
W = bsxfun(@times, W, 1.0 ./ sum(W,1));

end

function [rX, lambda, U, M, T, PI] = GibbsEst(samples, K, R, Nway, D, snum)
% each variable is estimated with samples mean

lambda = zeros(R,1);
U = cell(K,1);
M = cell(K,1);
T = cell(K,1);
PI = cell(K,1);

for k = 1 : K
    U{k} = zeros(Nway(k),R);
    M{k} = zeros(D(k),1);
    T{k} = zeros(D(k),1);
    PI{k} = zeros(D(k),1);
end

for n = 1 : snum
    lambda = lambda + samples(n).lambda;
    for k = 1 : K
        U{k} = U{k} + samples(n).U{k};
        M{k} = M{k} + samples(n).M{k};
        T{k} = T{k} + samples(n).T{k};
        PI{k} = PI{k} + samples(n).PI{k};
    end
end

lambda = lambda .* 1.0 ./ snum;
for k = 1 : K
    U{k} = U{k} .* 1.0 ./ snum;
    M{k} = M{k} .* 1.0 ./ snum;
    T{k} = T{k} .* 1.0 ./ snum;
    PI{k} = PI{k} .* 1.0 ./ snum;
end

rX = ktensor(lambda,U);

end