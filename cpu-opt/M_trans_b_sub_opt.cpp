#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <stdio.h>
#include "M_trans_b_opt.h"

#define B 16

void M_trans_b_sub_opt( double wPtr[], unsigned long int atomsPtr[], unsigned long int voxelsPtr[], unsigned long int fibersPtr[], double valuesPtr[], double DPtr[], double YPtr[], int nFibers, int nTheta, int nCoeffs )
{   
   int k;
   #pragma omp parallel for num_threads(num_of_cores)
   for (k = 0; k <= nCoeffs/B; k++) 
   {      
   	   int k1=k*B;
       int x[B],y[B],tmp;
       double val[B],tmp1;

       for (int i = 0; i < B; ++i)
       {
           x[i] =  atomsPtr[k1+i];
           y[i] =  voxelsPtr[k1+i];
       }
       
       for (int i = 0; i < B; ++i)
       {
           val[i]  = cblas_ddot (nTheta, (DPtr+x[i] ),  1, (YPtr+y[i]), 1);
       }

       for (int i = 0; i < B; ++i)
       {
           tmp = fibersPtr[k1+i];
           tmp1 = val[i]   *   valuesPtr[k1+i];
           #pragma omp atomic 
           wPtr[tmp] += tmp1;
       }
   }
   return;
}