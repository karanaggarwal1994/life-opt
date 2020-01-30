/* M_times_w.h

   ----------------------------------------------------------------------
   This file is part of LiFE toolbox

   Copyright (C) 2015 Cesar Caiafa & Franco Pestilli
   ----------------------------------------------------------------------
*/
#ifndef __M_W_OPT_H__
#define __M_W_OPT_H__

/* The entries in b are non-negative, those in d strictly positive */
void M_w_sub_opt( double YPtr[] , unsigned long int atomsPtr[], unsigned long int voxelsPtr[], unsigned long int fibersPtr[], double valuesPtr[], double DPtr[], double wPtr[], int nTheta, int nVoxels, int nCoeffs, unsigned long int vox[], int nvox);


#endif