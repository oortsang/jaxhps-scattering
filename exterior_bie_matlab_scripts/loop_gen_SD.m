% n_vals = [6 10 14 20]; % Order of Gauss panels
n_vals = [6 10 14 20 ]; % Order of Gauss panels
nref_vals = [ 1 2 3 4 5 ];  % Number of levels of uniform mesh refinement
% nref_vals = [ 8 ];
% rect = [-0.5 0.5 -0.5 0.5];
rect = [-1 1 -1 1];
zk = 20; % Wavenumber

for i=1:length(n_vals)
    n = n_vals(i);
    for j=1:length(nref_vals)
        nref = nref_vals(j);
        chnkr = squarechunker(n, nref, rect);
        Skern = kernel('helmholtz', 's', zk);
        Dkern = kernel('helmholtz', 'd', zk);
        S = chunkermat(chnkr, Skern);
        D = chunkermat(chnkr, Dkern);
        filename = sprintf('../data/wave_scattering/SD_matrices/SD_k%d_n%d_nside%d_dom%g.mat', zk, n, 2^nref, rect(2));
        % print update that we're saving a particular file
        disp(['Saving to ' filename])
        % S(1, 1:3)
        
        save(filename, 'S', 'D', '-v7.3')
    end
end
