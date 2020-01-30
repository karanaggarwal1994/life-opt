// BSD 3-Clause License
// 
// Copyright (c) 2019, Karan Aggarwal (karan@iisc.ac.in)
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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