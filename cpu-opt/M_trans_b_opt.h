/* Mtransp_times_b.h

   ----------------------------------------------------------------------
   This file is part of LiFE toolbox

   Copyright (C) 2015 Cesar Caiafa & Franco Pestilli
   ----------------------------------------------------------------------
*/
#ifndef __M_TRANS_B_OPT_H__
#define __M_TRANS_B_OPT_H__

/* The entries in b are non-negative, those in d strictly positive */
void M_trans_b_sub_opt( double wPtr[], unsigned long int atomsPtr[], unsigned long int voxelsPtr[], unsigned long int fibersPtr[], double valuesPtr[], double DPtr[], double YPtr[], int nFibers, int nTheta, int nCoeffs );


#endif