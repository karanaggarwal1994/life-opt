/* M_times_w.c

   ----------------------------------------------------------------------
   This file is part of LiFE toolbox

   Copyright (C) 2016 Cesar Caiafa & Franco Pestilli
   ----------------------------------------------------------------------
*/
#include "M_w_opt.h"
#include "mex.h"

/*Ax = M_times_w(
[0] - a1, 
[1] - a2 , 
[2] - a3 ,
[3] - A.Phi.vals,
[4] - A.DictSig,
[5] - x,
[6] - nTheta,
[7] - nVoxels)- 
[?] - b;*/
/* ----------------------------------------------------------------------- */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
/* ----------------------------------------------------------------------- */
{  const mxArray *atoms;
   const mxArray *voxels;
   const mxArray *fibers;
   const mxArray *values;
   const mxArray *D;
   const mxArray *w;
   const mxArray *vox;
   const mxArray *vox_size;
   mxArray *Y;
   
   
      
   int
       nVoxels, nTheta, nCoeffs,nvox;
   
   /*unsigned int   dims[2]; */
     
   /* Free memory and exit if no parameters are given */
   if (nrhs == 0)
   {  if (nlhs != 0) { mexErrMsgTxt("No output arguments expected."); }
      return ;
   }

   /* Check for proper number of arguments */
   // if (nrhs != 8)  { mexErrMsgTxt("Eight input arguments required."); }
   // if ((nlhs  > 1)               ) { mexErrMsgTxt("Too many output arguments.");             }

    /* Extract the arguments */
   atoms = (mxArray *)prhs[0];
   voxels = (mxArray *)prhs[1];
   fibers = (mxArray *)prhs[2];
   values = (mxArray *)prhs[3];
   D = (mxArray *)prhs[4];
   w = (mxArray *)prhs[5];
   nTheta = (int) mxGetScalar(prhs[6]);
   nVoxels= (int) mxGetScalar(prhs[7]);
   vox = (mxArray *)prhs[8];
   nvox= mxGetNumberOfElements(vox);
   
   Y = mxCreateNumericMatrix(nTheta * nVoxels, 1, mxDOUBLE_CLASS, mxREAL);
   
    /* Verify validity of input arguments */
    if (mxIsEmpty(atoms) || mxIsEmpty(voxels) || mxIsEmpty(fibers) || mxIsEmpty(values) || mxIsEmpty(D) || mxIsEmpty(w))
    {   /* If arguments are empty, simply return an empty weights */
        plhs[0] = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
        mexWarnMsgTxt("Returning empty weights.");
        return ;       
    }

    /* Verify validity of input argument 'atoms' */
    // if (atoms != NULL)
    // {  if (!mxIsDouble(atoms)  || ((mxGetM(atoms) > 1) &&
    //       (mxGetN(atoms) > 1)) || (mxGetNumberOfDimensions(atoms) != 2))
    //    {   mexErrMsgTxt("Parameter 'atoms' has to be a double vector.");
    //    }
    // }
   
    // /* Verify validity of input argument 'voxels' */
    // if (voxels != NULL)
    // {  if (!mxIsDouble(voxels)  || ((mxGetM(voxels) > 1) &&
    //       (mxGetN(voxels) > 1)) || (mxGetNumberOfDimensions(voxels) != 2))
    //    {   mexErrMsgTxt("Parameter 'voxels' has to be a double vector.");
    //    }
    // }
   
    // /* Verify validity of input argument 'fibers' */
    // if (fibers != NULL)
    // {  if (!mxIsDouble(fibers)  || ((mxGetM(fibers) > 1) &&
    //       (mxGetN(fibers) > 1)) || (mxGetNumberOfDimensions(fibers) != 2))
    //    {   mexErrMsgTxt("Parameter 'fibers' has to be a double vector.");
    //    }
    // }
      
    /* Verify validity of input argument 'values' */
    if (values != NULL)
    {  if (!mxIsDouble(values)  || ((mxGetM(values) > 1) &&
          (mxGetN(values) > 1)) || (mxGetNumberOfDimensions(values) != 2))
       {   mexErrMsgTxt("Parameter 'values' has to be a double vector.");
       }
    }   
   
    /* Verify validity of input argument 'D' */
    if (D != NULL)
    {  if (!mxIsDouble(D)  || ((mxGetM(D) > 1) &&
          (mxGetN(D) < 2)) || (mxGetNumberOfDimensions(D) != 2))
        {   mexErrMsgTxt("Parameter 'D' has to be a double matrix.");
        }
    }

    /* Verify validity of input argument 'Y' */
    if (w != NULL)
    {  if (!mxIsDouble(w)  || ((mxGetM(w) > 1) && (mxGetN(Y) > 1)) || (mxGetNumberOfDimensions(Y) != 2))
       {   mexErrMsgTxt("Parameter 'w' has to be a double vector.");
       }
    }   
   
   /* Verify all vectors have the same lenght */
   if ((mxGetNumberOfElements(atoms) != mxGetNumberOfElements(voxels)) || (mxGetNumberOfElements(atoms) != mxGetNumberOfElements(fibers)) || (mxGetNumberOfElements(atoms) != mxGetNumberOfElements(values)))
       {   mexErrMsgTxt("Parameters 'atoms', 'voxels', 'fibers' and 'values' have to be of equal length.");
       }


   /* Call the core subroutine */  
   /*nTheta = mxGetM(D); */
   nCoeffs = mxGetM(atoms);
    
    
   M_w_sub_opt( mxGetPr(Y),(unsigned long int *) mxGetPr(atoms),(unsigned long int *) mxGetPr(voxels), (unsigned long int *) mxGetPr(fibers), mxGetPr(values), mxGetPr(D), mxGetPr(w), nTheta, nVoxels, nCoeffs, (unsigned long int *) mxGetPr(vox),nvox );
   plhs[0] = Y;

    return ;
}

