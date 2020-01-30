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

function life_opt_demo(sys,BLAS_PATH)

if ~exist('vistaRootPath.m','file')
    disp('Vistasoft package either not installed or not on matlab path.')
    error('Please, download it from https://github.com/vistalab/vistasoft');
end

if ~exist('mbaComputeFibersOutliers','file')
    disp('ERROR: mba package either not installed or not on matlab path.')
    error('Please, download it from https://github.com/francopestilli/mba')
end

if ~exist('feDemoDataPath.m','file')
    disp('ERROR: demo dataset either not installed or not on matlab path.')
    error('Please, download it from http://purl.dlib.indiana.edu/iusw/data/2022/20995/Demo_Data_for_Multidimensional_Encoding_of_Brain_Connectomes.tar.gz ')
end

dwiFile       = fullfile(feDemoDataPath('STN','sub-FP','dwi'),'run01_fliprot_aligned_trilin.nii.gz');
dwiFileRepeat = fullfile(feDemoDataPath('STN','sub-FP','dwi'),'run02_fliprot_aligned_trilin.nii.gz');
t1File        = fullfile(feDemoDataPath('STN','sub-FP','anatomy'),  't1.nii.gz');

fgFileName = fullfile(feDemoDataPath('STN','sub-FP','tractography'),'run01_fliprot_aligned_trilin_csd_lmax10_wm_SD_PROB-NUM01-500000.tck');
feFileName = 'LiFE_build_model_demo_STN_FP_CSD_PROB';

L = 360;
Niter = 501;

fe = feConnectomeInit(dwiFile,fgFileName,feFileName,[],dwiFileRepeat,t1File,L,[1,0]);
fe = feSet(fe,'fit',feFitModel_opt(feGet(fe,'model'),feGet(fe,'dsigdemeaned'),'bbnnls',Niter,'preconditioner',sys,BLAS_PATH));

rmse1 = feGet(fe, 'total rmse');
rmse2 = feGetRep(fe, 'total rmse');
weights = feGet(fe, 'fiber weights');
wnorm = sum(weights);
nnz = sum(weights~=0);

fprintf('*************************************************\n');
fprintf('RMSE1 (train error): %f\n', rmse1);
fprintf('RMSE2 (cross validation error): %f\n', rmse2);
fprintf('wnorm (summed weights): %f\n', wnorm);
fprintf('nnz (number of non zero weihgts): %d\n', nnz);
fprintf('*************************************************\n');

end
