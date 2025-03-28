% author: Dan Fortunato
addpath ~/software/chunkie
addpath ~/software/surfacefun

%%
n = 14;
nref = 8;                   % Number of levels of uniform mesh refinement
% rect = [-0.5 0.5 -0.5 0.5]; % Bounding box of compactly supported region
rect = [-1 1 -1 1];
zk = 100;

chnkr = squarechunker(n, nref, rect);
Skern = kernel('helmholtz', 's', zk);
Dkern = kernel('helmholtz', 'd', zk);
S = chunkermat(chnkr, Skern);
D = chunkermat(chnkr, Dkern);

%% Save
filename = sprintf('../data/SD_k%d_n%d_nside%d_dom=%g.mat', zk, n, 2^nref, rect(2));
save(filename, 'S', 'D', '-v7.3')

%% Plot

clf
hold on
for k = 1:chnkr.nch
    plot(chnkr.r(1,:,k), chnkr.r(2,:,k), '-o')
    drawnow
    shg
end