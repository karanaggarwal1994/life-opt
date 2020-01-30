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

function checkMexCompiled_cpu(varargin)
    source_file{1} = varargin{end-2};
    source_file{2} = varargin{end-1};
    BLAS_PATH = varargin{end};

    % Check input filename
    assert(ischar(source_file{1}),'source_file must be a string')
    assert(ischar(source_file{2}),'source_file must be a string')

    % Check extension is specified
    assert(~isempty(strfind(source_file{1},'.')),'source_file: no file extension specified')
    assert(~isempty(strfind(source_file{2},'.')),'source_file: no file extension specified')

    % Locate source file
    [pathstr{1},name{1},ext{1}] = fileparts(which(source_file{1}));
    [pathstr{2},name{2},ext{2}] = fileparts(which(source_file{2}));

    filename{1} = [pathstr{1} filesep name{1} ext{1}]; % Create filename
    mexfilename = [pathstr{1} filesep name{1} '.' mexext]; % Deduce mex file name based on current platform

    if strcmp(pathstr{1},'') || strcmp(pathstr{2},'')% source file not found
        error([source_file ': not found'])
    elseif exist(mexfilename,'file')~=3 || get_mod_date(mexfilename)<get_mod_date(filename{1})
         % if source file does not exist or it was modified after the mex file
        disp(['Compiling "' name ext '".'])
        d = cd;
        cd(pathstr{1})
        % compile, with options if appropriate
        try
            if length(varargin)>1
                options = varargin{1:end-2};
                mex(options,source_file{1},source_file{2})
            else
                mex(source_file,source_file{1},source_file{2})                
            end
             fprintf('Function %s successfuly compiled\n',source_file{1});
        catch lasterr
            prompt='Enter OpenBLAS path [ex. /home/ubuntu/OpenBLAS]:';
%             OpenBLAS_PATH = input(prompt,'s');
            OpenBLAS_PATH = BLAS_PATH;
            if isempty(OpenBLAS_PATH)     
                fprintf('ERROR: Could not compile function %s. \n',source_file{1});
                error('Please, install an appropriate compiler and run the scipts again (see http://www.mathworks.com/support/compilers). Mex files will be generated automatically. \n');
            else
                try
                    num_cores = feature('numCores');
                    lib = strcat('-L',OpenBLAS_PATH,'/lib/');
                    include = strcat('-I',OpenBLAS_PATH,'/include/');
                    cflags = 'CFLAGS="\$CFLAGS -fopenmp -O3 -lopenblas"';
                    ldflags = 'LDFLAGS="\$LDFLAGS -fopenmp -O3 -lopenblas"';
                    num_core = strcat('-Dnum_of_cores=',num2str(num_cores));
                    mex('-g','-largeArrayDims',lib,include,cflags,num_core,ldflags,source_file{1},source_file{2});
                catch lasterr
                    fprintf('ERROR: Could not compile function %s. \n',source_file{1});
                    error('Please, install an appropriate compiler and run the scipts again (see http://www.mathworks.com/support/compilers).');
                end
            end
            
            
        end
        
        cd(d)
    end

end

function datenum = get_mod_date(file)
%GET_MOD_DATE get file modified date

    d = dir(file);
    datenum = d.datenum;

end
