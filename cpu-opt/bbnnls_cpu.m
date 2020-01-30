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

function out = bbnnls_cpu(M, b, x0, opt, BLAS_PATH)
    % Few Initializations
    nCoeffs    = size(M.Phi.subs(:,1),1);
    nTheta     = size(M.DictSig,1);
    
    %Basic Compiler Optimizations
    Mg.atoms1   = uint64((M.Phi.subs(:,1)-1)*nTheta);
    Mg.voxels1  = uint64((M.Phi.subs(:,2)-1)*nTheta);
    Mg.fibers1  = uint64(M.Phi.subs(:,3)-1);
    Mg.vals1    = M.Phi.vals;
    
    %Data Restructuring
    [Mg.voxels2,order] = sort(Mg.voxels1);
    Mg.atoms2   = Mg.atoms1(order);
    Mg.fibers2  = Mg.fibers1(order);
    Mg.vals2    = Mg.vals1(order);
    
    [~, vox, ~] = unique(Mg.voxels2);
    vox = uint64(vox - 1);
    vox=vox(:);
    vox(size(vox)+1) = nCoeffs;
    
    % Assert condition to check parallelization constraint
    nThreads = feature('numCores');
    [~,maxmode]  = mode(Mg.voxels2);
    block_size = uint64(nCoeffs/nThreads);
    block_size = block_size(1);
    assert((block_size> maxmode), 'Code Paralleliztion is not possible!!!');
    
    % Computation Splitting
    [~, vox, ~] = unique(Mg.voxels2);
    vox = (uint64(vox - 1));
    vox=vox(:);
    vox(size(vox)+1)=nCoeffs;
    Mg.vox=vox;
    
    
    %Remove data dependence
%     tile_end_array
    tile = zeros([nThreads 1]);
    for i=2:nThreads
        b_size=(i-1)*block_size;
        [x y z] = find(vox>b_size,1);
        left = b_size - vox(x-1);
        right = vox(x)-b_size;
        if left<right
            tile(i)= vox(x-1);
        else
            tile(i)= vox(x);
        end
    end
    tile(nThreads+1)=nCoeffs(1);
    Mg.tile=uint64(tile(:));
    
    
    
    atoms1 = Mg.atoms1;
    voxels1= Mg.voxels1;
    fibers1= Mg.fibers1;
    vals1  = Mg.vals1;
    atoms2 = Mg.atoms2;
    voxels2= Mg.voxels2;
    fibers2= Mg.fibers2;
    vals2  = Mg.vals2;
    dptr  = M.DictSig;
    tile  = Mg.tile;
    ntile = size(Mg.tile,1);
    nvox = size(vox,1);
    nVoxels = size(M.Phi,2);
    
    
    
    
    
    
    % Compile C++ file to generate .mexa64 code;
    cpu_compile(BLAS_PATH);
    
    fgx = @(x) funcGrad(M, b, x ,Mg);
    % do some initialization for maintaining statistics
    out.iter = 0;
    out.iterTimes = nan*ones(opt.maxit,1);
    out.objTimes  = nan*ones(opt.maxit,1);
    out.pgTimes   = nan*ones(opt.maxit,1);
    out.trueError = nan*ones(opt.maxit,1);
    out.startTime = tic;
    out.status = 'Failure';

    % HINT: Very important for overall speed is to have a good x0
    out.x      = x0;
    out.refx   = x0;
    [out.refobj, out.grad]   = fgx(out.x);
    out.oldg   = out.grad;
    out.refg   = out.oldg;

    %% Begin the main algorithm
    if (opt.verbose)
        fprintf('Running: **** SBB-NNLS ****\n\n');
        fprintf('Iter   \t     Obj\t\t  ||pg||_inf\t\t ||x-x*||\n');
        fprintf('-------------------------------------------------------\n');
    end
    
    objectives = zeros(opt.maxit,1);
    while 1
        out.iter = out.iter + 1;
        [termReason, out.pgTimes(out.iter)] = checkTermination(opt, out);
        if (termReason > 0), break; end
        [step, out] = computeBBStep(M, b, out,Mg);
        out.x = out.x - step * out.grad;
        out.oldg = out.grad;
        out.x(out.x < 0) = 0;
        [out.obj, out.grad] =  funcGrad(M,b,out.x,Mg);
        objectives(out.iter) = out.obj;
        out.objTimes (out.iter) = out.obj;
        out.iterTimes(out.iter) = toc(out.startTime);
        

        if (opt.truex), out.trueError(out.iter) = norm(opt.xt-out.x); end
        if (opt.verbose)
             fprintf('%04d\t %E\t%E\t%E\n', out.iter, out.obj, out.pgTimes(out.iter), out.trueError(out.iter)); 
        end
    end % of while

    %%  Final statistics and wrap up
    out.time = toc(out.startTime);
    out.status = 'Success';
    out.termReason = setTermReason(termReason);
end

% Compute BB step; for SBB also modifies out.oldg, and this change must be
% passed back to the calling routine, else it will fail!
  function [step, out] = computeBBStep(A, ~, out, Mg)
    [nTheta]  = size(A.DictSig,1);  
    [nFibers] = size(A.Phi,3);
    [nVoxels] = size(A.Phi,2);      
    
    gp = find(out.x == 0 & out.grad > 0);
    out.oldg(gp) = 0;

    atoms = Mg.atoms2;
    voxels= Mg.voxels2;
    fibers= Mg.fibers2;
    vals  = Mg.vals2;
    dptr  = A.DictSig;
    wptr   = out.oldg;
    yptr   = zeros(nTheta , nVoxels);
    tile  = Mg.tile;
    tile_size = size(Mg.tile,1);
    [nCoeffs] = size(Mg.atoms2(:));
    Ag = M_w_opt(Mg.atoms2 , Mg.voxels2 , Mg.fibers2 , Mg.vals2, A.DictSig,out.oldg,nTheta,nVoxels,Mg.tile,size(Mg.tile,1));
    if (mod(out.iter, 2) == 0) 
        step = (out.oldg' * out.oldg) / (Ag' * Ag);
    else
        numer = Ag' * Ag ;
        Ag = M_trans_b_opt(Mg.atoms1 ,Mg.voxels1 ,Mg.fibers1, Mg.vals1, A.DictSig,reshape(Ag,[nTheta,nVoxels]),nFibers); % MEX Intel compiler version
        Ag(gp) = 0;
       step = numer / (Ag' * Ag);
    end
end

function [f, g] = funcGrad(A, b, x, Mg) 
    [nFibers] = size(A.Phi,3);
    [nTheta]  = size(A.DictSig,1); 
    [nVoxels] = size(A.Phi,2);   
    Ax = M_w_opt(Mg.atoms2 , Mg.voxels2 , Mg.fibers2 , Mg.vals2,A.DictSig,x,nTheta,nVoxels,Mg.tile,size(Mg.tile,1))-b;
    f = 0.5 * norm(Ax)^2;
    if (nargout > 1)
    g = M_trans_b_opt(Mg.atoms1 ,Mg.voxels1 ,Mg.fibers1, Mg.vals1, A.DictSig,reshape(Ax,[nTheta,nVoxels]),nFibers);% MEX Intel compiler version     
    end
end

function cpu_compile(BLAS_PATH)
    checkMexCompiled_cpu('-largeArrayDims', '-output', 'compute_diag', '-DNDEBUG', 'compute_diag.c', 'compute_diag_sub.c',BLAS_PATH)
    checkMexCompiled_cpu('-largeArrayDims', '-output', 'M_times_w', '-DNDEBUG', 'M_times_w.c', 'M_times_w_sub.c',BLAS_PATH)
    checkMexCompiled_cpu('-largeArrayDims', '-output', 'Mtransp_times_b', '-DNDEBUG','Mtransp_times_b.c', 'Mtransp_times_b_sub.c',BLAS_PATH)
    checkMexCompiled_cpu('-largeArrayDims', '-output', 'M_w_opt', '-DNDEBUG', 'M_w_opt.cpp', 'M_w_sub_opt.cpp',BLAS_PATH)
    checkMexCompiled_cpu('-largeArrayDims', '-output', 'M_trans_b_opt', '-DNDEBUG','M_trans_b_opt.cpp', 'M_trans_b_sub_opt.cpp',BLAS_PATH)
    
end

% check various termination criteria; return norm of pg
% the strictest is norm of pg
function [v, pg] = checkTermination(options, out)
    % pgnorm limit -- need to check this first of all
    gp = find( (out.x ~= 0 | out.grad < 0));

    pg = norm(out.grad(gp), 'inf');
    if (pg < options.tolg), v=8; return; end

    % First check if we are doing termination based on running time
    if (options.time_limit)
        out.time = etime(clock, out.start_time);
        if (out.time >= options.maxtime)
            v = 1;
            return;
        end
    end

    % Now check if we are doing break by tolx
    if (options.use_tolx)
        if (norm(out.x-out.oldx)/norm(out.oldx) < options.tolx)
            v = 2;
            return;
        end
    end

    % Are we doing break by tolo (tol obj val)
    if (options.use_tolo && out.iter > 2)
        delta = abs(out.objTimes(out.iter-1)-out.objTimes(out.iter-2));
        if (delta < options.tolo)
            v = 3;
            return;
        end
    end

    % Finally the plain old check if max iter has been achieved
    if (out.iter >= options.maxit)
        v = 4;
        return;
    end

    % KKT violation
    if (options.use_kkt)
        if abs(out.x' * out.grad) <= options.tolk
            v = 7;
            return;
        end
    end


    % All is ok...
    v = 0;
end

%% Prints status
function showStatus(out, options)
    if (options.verbose)
        fprintf('.');
        if (mod(out.iter, 30) == 0)
            fprintf('\n');
        end
    end
end

% String representation of termination
function r = setTermReason(t)
    switch t
      case 1
        r = 'Exceeded time limit';
      case 2
        r = 'Relative change in x small enough';
      case 3
        r = 'Relative change in objvalue small enough';
      case 4
        r = 'Maximum number of iterations reached';
      case 5
        r = '|x_t+1 - x_t|=0 or |grad_t+1 - grad_t| < 1e-9';
      case 6
        r = 'Line search failed';
      case 7
        r = '|x^T * grad| < opt.pbb_gradient_norm';
      case 8
        r = '|| grad ||_inf < opt.tolg';
      case 100
        r = 'The active set converged';
      otherwise
        r = 'Undefined';
    end
end

