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

function out = bbnnls_gpu(M, b, x0, opt)
    % Check whether GPU is available and installed correctly
    assert(gpuDeviceCount>0,'No GPU device found!!!');

    % Few Initializations
    nCoeffs    = size(M.Phi.subs(:,1));
    nTheta     = size(M.DictSig,1);
    
    Mg.nCoeffs = nCoeffs(:,1);
    Mg.nTheta  = size(M.DictSig,1);
    Mg.nAtoms  = size(M.Phi,1);  
    Mg.nFibers = size(M.Phi,3);
    Mg.nVoxels = size(M.Phi,2);  
    
    % Padding Code here
    nTh_mod = mod(size(M.DictSig,1),32);
    if nTh_mod ~=0
        pad =32-nTh_mod; 
        M.DictSig = padarray(M.DicSig,[pad 0],'post');
        b = reshape(b,[Mg.nTheta Mg.nVoxels]);
        b = padarray(b,[gather(pad) 0],'post' );
        b = b(:);
        Mg.nTheta  = size(M.DictSig,1);
        nTheta = Mg.nTheta;
    end
    
    % ***Basic Compiler Optimizations + Data Restructuring***
    
    % *Atom-based data restructuring
    atom_ds.atoms   = uint64((M.Phi.subs(:,1)-1)*nTheta);
    atom_ds.voxels  = uint64((M.Phi.subs(:,2)-1)*nTheta);
    atom_ds.fibers  = uint64(M.Phi.subs(:,3)-1);
    atom_ds.vals    = M.Phi.vals;
    
    % *Voxel-based data restructuring
    [vox_ds.voxels,order] = sort(atom_ds.voxels);
    vox_ds.atoms          = atom_ds.atoms(order);
    vox_ds.fibers         = atom_ds.fibers(order);
    vox_ds.vals           = atom_ds.vals(order);
    
    % ***Computation Partitioning***
    
    % Coefficient-based partitioning
    coeff = uint64((0:nCoeffs));
    coeff = coeff';
    
    % Voxel-based partitioning
    [~, vox, ~] = unique(vox_ds.voxels);
    vox = uint64(vox - 1);
    vox = vox(:);
    vox(size(vox)+1) = nCoeffs;
    
    % Atom-based partitioning
    [~, atom, ~] = unique(atom_ds.atoms);
    atom = uint64(atom - 1);
    atom = atom(:);
    atom(size(atom)+1) = nCoeffs;

    % ***Setting the parameters***
    
    % Computation Partitioning
    Mg.cp_mw = vox;   %[valid options: vox,coeff]
    Mg.cp_my = coeff; %[valid options: vox,atom,coeff]
    
    % Data restructuring
    ds_mw  = vox_ds;  %[valid options: vox_ds]
    ds_my  = atom_ds; %[valid options: atom_ds,vox_ds]
    
    % Multiple Computations
    mc_mw = 4;
    mc_my = 4;
    
    % Setting the choice to run a code section at runtime
    Mg.choice_mw=uint64((size(Mg.cp_mw,1)-1)==(Mg.nCoeffs));
    Mg.choice_my=uint64((size(Mg.cp_my,1)-1)==(Mg.nCoeffs));
    
    % GPU arrays support only fundamental numeric or logical data types.
    % Hence, converting them Mg to GPU arrays format
    Mg.mw_atoms  = ds_mw.atoms;
    Mg.mw_voxels = ds_mw.voxels;
    Mg.mw_fibers = ds_mw.fibers;
    Mg.mw_vals   = ds_mw.vals;
    
    Mg.my_atoms  = ds_my.atoms;
    Mg.my_voxels = ds_my.voxels;
    Mg.my_fibers = ds_my.fibers;
    Mg.my_vals   = ds_my.vals;
    
    % Compile C file to generate .mex code
    cpu_compile();
    
    % Compile CUDA file to generate .PTX code
    gpu_compile(nTheta,mc_mw,mc_my);
    
    % Set default GPU parameters
    MTW_CU_OBJ = parallel.gpu.CUDAKernel('gpu_opt_code.ptx','gpu_opt_code.cu','M_times_w');        
    MTW_CU_OBJ.GridSize        = [ceil(size(Mg.cp_mw)/mc_mw) 1];
    MTW_CU_OBJ.ThreadBlockSize = [(32*mc_mw) 1];
 
    MTB_CU_OBJ= parallel.gpu.CUDAKernel('gpu_opt_code.ptx','gpu_opt_code.cu','Mtransp_times_b');
    MTB_CU_OBJ.GridSize         = [ceil(size(Mg.cp_my)/mc_my) 1];
    MTB_CU_OBJ.ThreadBlockSize  = [(32*mc_my) 1];
    
    Mg.Nphi    = M.Nphi;
    Mg.Ntheta  = M.Ntheta;
    Mg.orient  = M.orient;
    Mg.DictSig = M.DictSig;
    Mg         = structfun(@gpuArray,Mg,'UniformOutput',false);
    b          = gpuArray(b);
    x0         = gpuArray(x0);
    clear order;
    
    [out.obj,out.grad] =  funcGrad(Mg,b,x0,MTW_CU_OBJ,MTB_CU_OBJ);
    
    % do some initialization for maintaining statistics
    out.iter      = 0;
    out.iterTimes = nan*ones(opt.maxit,1,'gpuArray');
    out.objTimes  = nan*ones(opt.maxit,1,'gpuArray');
    out.pgTimes   = nan*ones(opt.maxit,1,'gpuArray');
    out.trueError = nan*ones(opt.maxit,1,'gpuArray');
    out.startTime = tic;
    out.status    = 'Failure';

    % HINT: Very important for overall speed is to have a good x0
    out.x      = x0;
    out.refx   = x0;
    [out.refobj, out.grad] = funcGrad(Mg,b,out.x,MTW_CU_OBJ,MTB_CU_OBJ);
    out.oldg   = out.grad;
    out.refg   = out.oldg;

    %% Begin the main algorithm
    if (opt.verbose)
       fprintf('Running: **** SBB-NNLS ****\n\n');
       fprintf('Iter   \t     Obj\t\t  ||pg||_inf\t\t ||x-x*||\n');
       fprintf('-------------------------------------------------------\n');
    end
    objectives = zeros(opt.maxit,1,'gpuArray');
    while 1
        out.iter = out.iter + 1;
        [termReason, out.pgTimes(out.iter)] = checkTermination(opt, out);
        if (termReason > 0), break; end
            [step, out] = computeBBStep(Mg,b,out,MTW_CU_OBJ,MTB_CU_OBJ);
            out.x       = out.x - step * out.grad;
            out.oldg    = out.grad;        
            out.x(out.x < 0) = 0;        
            [out.obj,out.grad] =  funcGrad(Mg,b,out.x,MTW_CU_OBJ,MTB_CU_OBJ);
            objectives(out.iter) = out.obj;
            out.objTimes (out.iter) = out.obj;
            out.iterTimes(out.iter) = toc(out.startTime);
            if (opt.truex), out.trueError(out.iter) = norm(opt.xt-out.x);             
            end
            if (opt.verbose)
                fprintf('%04d\t %E\t%E\t%E\n', out.iter, out.obj, out.pgTimes(out.iter), out.trueError(out.iter)); 
            end
     end 
    
    %%  Final statistics and wrap up
    out.time   = toc(out.startTime);
    out.status = 'Success';
    out.termReason = setTermReason(termReason);
    out.x    = gather(out.x);
    out.refx = gather(out.refx);
    out.grad = gather(out.grad);
    out.oldg = gather(out.oldg);
    out.refg = gather(out.refg);
end

% Compute BB step; for SBB also modifies out.oldg, and this change must be
% passed back to the calling routine, else it will fail!
function [step, out] = computeBBStep(Mg,~,out,MTW_CU_OBJ,MTB_CU_OBJ)
    [nTheta]  = Mg.nTheta; 
    [nFibers] = Mg.nFibers;
    [nVoxels] = Mg.nVoxels;
    [nCoeffs] = Mg.nCoeffs;
    
    gp = find(out.x == 0 & out.grad > 0);
    out.oldg(gp) = 0;

     D       =  gpuArray(Mg.DictSig);
     D_vec   =  D(:);
     Y       =  zeros(nTheta,nVoxels,'gpuArray');
     Y_vec   =  Y(:);
     Ag      =  feval(MTW_CU_OBJ,Y_vec,Mg.mw_atoms,Mg.mw_voxels,Mg.mw_fibers,Mg.mw_vals,D_vec,out.oldg,nTheta,nVoxels,nCoeffs,Mg.cp_mw,size(Mg.cp_mw,1),Mg.choice_mw);
     wait(gpuDevice);
     if (mod(out.iter, 2) == 0)
         step = (out.oldg' * out.oldg) / (Ag' * Ag);
     else
         numer = Ag' * Ag;
         w       =  zeros(nFibers,1,'gpuArray');
         Ag      =  feval(MTB_CU_OBJ,w,Mg.my_atoms,Mg.my_voxels,Mg.my_fibers,Mg.my_vals,D_vec,Ag,nFibers,nTheta,nCoeffs,Mg.cp_my,size(Mg.cp_my,1),Mg.choice_my);
         wait(gpuDevice);
         Ag(gp) = 0;        
         step = numer / (Ag' * Ag);
    end
end

% compute obj function and gradient --- requires good implementation of A*x
% and A'*y for appropriate x and y
function [f,g] = funcGrad(Mg,b,x,MTW_CU_OBJ,MTB_CU_OBJ)
    [nFibers] = Mg.nFibers; 
    [nTheta]  = size(Mg.DictSig,1);
    [nVoxels] = Mg.nVoxels;   
     nCoeffs  = Mg.nCoeffs;
   
         D       =  gpuArray(Mg.DictSig);
         D_vec   =  D(:);
         Y       =  zeros(nTheta,nVoxels,'gpuArray');
         Y_vec   =  Y(:);         
         Ax      =  feval(MTW_CU_OBJ,Y_vec,Mg.mw_atoms,Mg.mw_voxels,Mg.mw_fibers,Mg.mw_vals,D_vec,x,nTheta,nVoxels,nCoeffs,Mg.cp_mw,size(Mg.cp_mw,1),Mg.choice_mw);
         wait(gpuDevice);
         Ax      =  Ax-b;
         
         if (nargout > 1)
            w       = zeros(nFibers,1,'gpuArray');
            g = feval(MTB_CU_OBJ,w,Mg.my_atoms,Mg.my_voxels,Mg.my_fibers,Mg.my_vals,D_vec,Ax,nFibers,nTheta,nCoeffs,Mg.cp_my,size(Mg.cp_my,1),Mg.choice_my);  
            wait(gpuDevice);
         end
         f       =  0.5*norm(Ax)^2;   
end

function cpu_compile()
    checkMexCompiled('-largeArrayDims', '-output', 'compute_diag', '-DNDEBUG', 'compute_diag.c', 'compute_diag_sub.c')
    checkMexCompiled('-largeArrayDims', '-output', 'M_times_w', '-DNDEBUG', 'M_times_w.c', 'M_times_w_sub.c')
    checkMexCompiled('-largeArrayDims', '-output', 'Mtransp_times_b', '-DNDEBUG','Mtransp_times_b.c', 'Mtransp_times_b_sub.c')
end

function gpu_compile(nTheta,nc_mw,nc_my)
    cudafilename = strcat('gpu_opt_code', '.cu');  
    [pathstr{1},name{1},ext{1}] = fileparts(which(cudafilename));
    cudafilename = [pathstr{1} filesep name{1} ext{1}];
    ptxfilename = [pathstr{1} filesep name{1} '.ptx'];
    cmdline = sprintf('nvcc -ptx %s --output-file %s -DTheta=%d -Dnc_mw=%d -Dnc_my=%d', cudafilename,ptxfilename,nTheta,nc_mw,nc_my);
    system(cmdline);
end

% check various termination criteria; return norm of pg
% the strictest is norm of pg
function [v,pg] = checkTermination(options, out)
    % pgnorm limit -- need to check this first of all
    gp = find( (out.x ~= 0 | out.grad < 0));
    pg = norm(out.grad(gp), 'inf');
    temp = out.grad(gp);
    
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

