% This version was derived from the original version by Pestilli and Caiafa.
% Copyright (2015), Franco Pestilli (Indiana Univ.) - Cesar F. Caiafa (CONICET)
% email: pestillifranco@gmail.com and ccaiafa@gmail.com
% https://github.com/francopestilli/life available under
% https://github.com/francopestilli/life/blob/master/LICENSE.md
% 
% BSD 3-Clause License
% 
% Copyright (c) 2019, Karan Aggarwal (karan@iisc.ac.in)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its
%    contributors may be used to endorse or promote products derived from
%    this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function [fit, w, R2] = feFitModel_opt(varargin)
M = varargin{1};
dSig = varargin{2};
fitMethod = varargin{3};
Niter = varargin{4};
preconditioner = varargin{5};
sys = varargin{5};
BLAS_PATH = varargin{5};


% feFitModel() function in LiFE but restricted to the
% 
% BBNNLS algorithm and using the Factorization model.
% M is the factorization model composed by:
%
%   M.DictSig    Dictionary
%   
% Below are the old comments which are not longer correct in general
% Fit the LiFE model.
%
% Finds the weights for each fiber to best predict the directional
% diffusion signal (dSig)
%
%  fit = feFitModel(M,dSig,fitMethod)
%
% dSig:  The diffusion weighted signal measured at each
%        voxel in each direction. These are extracted from 
%        the dwi data at some white-matter coordinates.
% M:     The LiFE difusion model matrix, constructed
%        by feConnectomeBuildModel.m
%
% fitMethod: 
%  - 'bbnnls' - DEFAULT and best, faster large-scale solver.
%
% See also: feCreate.m, feConnectomeBuildModel.m, feGet.m, feSet.m
%
% Example:
%
% Notes about the LiFE model:
%
% The rows of the M matrix are nVoxels*nBvecs. We are going to predict the
% diffusion signal in each voxel for each direction.
%
% The columns of the M matrix are nFibers + nVoxels.  The diffusion signal
% for each voxel is predicted as the weighted sum of predictions from each
% fibers that passes through a voxel plus an isotropic (CSF) term.
%
% In addition to M, we typically return dSig, which is the signal measured
% at each voxel in each direction.  These are extracted from the dwi data
% and knowledge of the roiCoords.

if nargin <6
    [nFibers] = size(M.Phi,3);
    w0 = zeros(nFibers,1);
else
    w0 = varargin{6};
end


if strcmp(preconditioner,'preconditioner') 
    [nFibers] = size(M.Phi,3); %feGet(fe,'nfibers');
    checkMexCompiled('-largeArrayDims', '-output', 'compute_diag', '-DNDEBUG', 'compute_diag.c', 'compute_diag_sub.c')
    h = compute_diag(M.Phi.subs(:,1), M.Phi.subs(:,3), M.Phi.vals, M.DictSig,nFibers);
    vals = M.Phi.vals./h(M.Phi.subs(:,3));
    M.Phi = sptensor(M.Phi.subs,vals,size(M.Phi));
end


switch fitMethod
   case {'bbnnls'}   
    tic
    fprintf('\nLiFE: Computing least-square minimization with BBNNLS...\n')
    opt = solopt;
    opt.maxit = Niter;
    opt.use_tolo = 1;
    opt.tolg = 1e-5;
    opt.verbose = 1;

    if sys==0
        out_data = bbnnls(M,dSig,w0,opt);
    elseif sys==1
        out_data = bbnnls_cpu(M,dSig,w0,opt,BLAS_PATH);
    elseif sys==2
        out_data = bbnnls_gpu(M,dSig,w0,opt);

    end

    if strcmp(preconditioner,'preconditioner')
        out_data.x = out_data.x./h;
    end
    
    fprintf('BBNNLS status: %s\nReason: %s\n',out_data.status,out_data.termReason);
    w = out_data.x;
    curr=toc;
    
    fprintf('*******************************************************************************\n');
    fprintf(' SBBNNLS process completed in..................................... %2.3fminutes\n',out_data.iterTimes(500)/60)
    fprintf(' Total overhead for the SBBNNLS process completed in ............. %2.3fminutes\n',(curr-out_data.iterTimes(500))/60)
    fprintf(' Total time for the SBBNNLS process (with overhead) completed in.. %2.3fminutes\n',curr/60)
    fprintf('*******************************************************************************\n');
    
    % Save the state of the random generator so that the stochasit cfit can be recomputed.
    defaultStream = RandStream.getGlobalStream; %RandStream.getDefaultStream;
    fit.randState = defaultStream.State;   
    
    % Save out some results 
    fit.results.R2        = [];
    fit.results.nParams   = size(M,2);
    fit.results.nMeasures = size(M,1);
    R2=[];

   otherwise
     error('Cannot fit LiFE model using method: %s.\n',fitMethod);
end

% Save output structure.
fit.weights             = w;
fit.params.fitMethod    = fitMethod;

end
