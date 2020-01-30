#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include "M_w_opt.h"
#include <iostream>
// #define T 16

void M_w_sub_opt( double YPtr[], unsigned long int atomsPtr[], unsigned long int voxelsPtr[], unsigned long int fibersPtr[], double valuesPtr[], double DPtr[], double wPtr[], int nTheta, int nVoxels, int nCoeffs, unsigned long int tile[], int ntile)
{   
    #pragma omp parallel for num_threads(num_of_cores)
    for (int i = 0; i < ntile-1; ++i)
    {
      for(int k=tile[i];k<tile[i+1];k++){     
        int x=atomsPtr[k];
        int y=voxelsPtr[k];
        int z=fibersPtr[k];
        double v = wPtr[z]*valuesPtr[k];
        cblas_daxpy (nTheta, v,  (DPtr+x),  1, (YPtr+y), 1);
      }
    }
    return;
}