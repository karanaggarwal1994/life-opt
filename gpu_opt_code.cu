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

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#define FULL_MASK 0xffffffff

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val) {
  unsigned long long *address_as_ull = (unsigned long long *)address;
  unsigned long long old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

__global__ void compute_diag_sub(double *dPtr, const unsigned long *atomsPtr,
                                 const unsigned long *fibersPtr,
                                 const double *valuesPtr, const double *DPtr,
                                 const unsigned long nFibers, const int nTheta,
                                 const unsigned long nCoeffs) {

  unsigned long k = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long offset = 0;
  unsigned long stride = gridDim.x * blockDim.x;
  while ((k + offset) < nCoeffs) {
    double val = 0;
    int atom_index = atomsPtr[k + offset];
    for (int i = 0; i < nTheta; i++) {
      val += DPtr[atom_index + i] * DPtr[atom_index + i];
    }
    val = val * valuesPtr[k + offset] * valuesPtr[k + offset];
    atomicAdd(&dPtr[fibersPtr[k + offset]], val);
    offset += stride;
  }
  return;
}

__global__ void M_times_w(double *YPtr, const unsigned long *atomsPtr,
                          const unsigned long *voxelsPtr,
                          const unsigned long *fibersPtr,
                          const double *valuesPtr, const double *DPtr,
                          const double *wPtr, const int nTheta,
                          const unsigned long nVoxels,
                          const unsigned long nCoeffs, const unsigned long *vox,
                          const long nvox, int ch) {
  unsigned long long k = (threadIdx.x / 32) + (blockIdx.x * nc_mw);
  if (k < nvox) {
    if (ch == 0) {
      unsigned long voxel_index = voxelsPtr[vox[k]];
      __shared__ double y[nc_mw][Theta];
      int th_id = threadIdx.x % 32;
      while (th_id < nTheta) {
        y[threadIdx.x / 32][th_id] = YPtr[voxel_index + th_id];
        th_id = th_id + 32;
      }
      __syncwarp();
#pragma unroll 8
      for (int t = vox[k]; t < vox[k + 1]; t++) {
        unsigned long fiber_index = fibersPtr[t];
        unsigned long atom_index = atomsPtr[t];
        if (wPtr[fiber_index]) {
          th_id = threadIdx.x % 32;
          double val = wPtr[fiber_index] * valuesPtr[t];
          while (th_id < nTheta) {
            y[threadIdx.x / 32][th_id] += DPtr[atom_index + th_id] * val;
            th_id = th_id + 32;
          }
        }
        __syncwarp();
      }
      __syncwarp();
      th_id = threadIdx.x % 32;
      while (th_id < nTheta) {
        YPtr[voxel_index + th_id] = y[threadIdx.x / 32][th_id];
        th_id = th_id + 32;
      }
    } else {
      unsigned long voxel_index = voxelsPtr[k];
      unsigned long fiber_index = fibersPtr[k];
      unsigned long atom_index = atomsPtr[k];

      int th_id = threadIdx.x % 32;
      if (wPtr[fiber_index]) {
        double val = wPtr[fiber_index] * valuesPtr[k];
        while (th_id < nTheta) {
          atomicAdd(&YPtr[voxel_index + th_id], DPtr[atom_index + th_id] * val);
          th_id = th_id + 32;
        }
      }
    }
  }
  return;
}

__global__ void Mtransp_times_b(
    double *wPtr, const unsigned long *atomsPtr, const unsigned long *voxelsPtr,
    const unsigned long *fibersPtr, const double *valuesPtr, const double *DPtr,
    const double *YPtr, const unsigned long nFibers, const int nTheta,
    const long nCoeffs, const unsigned long *vox, const long nvox, int ch) {
  unsigned long long k = (threadIdx.x / 32) + (blockIdx.x * nc_my);
  if (k < nvox) {
    if (ch == 0) {
      for (int t = vox[k]; t < vox[k + 1]; t++) {
        unsigned long voxel_index = voxelsPtr[t];
        unsigned long atom_index = atomsPtr[t];
        unsigned long fiber_index = fibersPtr[t];

        double val = 0;
        int th_id = threadIdx.x % 32;
        while (th_id < nTheta) {
          val = val + (DPtr[atom_index + th_id] * YPtr[voxel_index + th_id]);
          th_id = th_id + 32;
        }
        __syncwarp();
#pragma unroll 5
        for (int j = 16; j >= 1; j = j / 2) {
          val += __shfl_down_sync(FULL_MASK, val, j);
        }
        __syncwarp();
        if ((threadIdx.x % 32) == 0) {
          atomicAdd(&wPtr[fiber_index], val * valuesPtr[t]);
        }
        __syncwarp();
      }
    } else {
      unsigned long voxel_index = voxelsPtr[k];
      unsigned long atom_index = atomsPtr[k];
      unsigned long fiber_index = fibersPtr[k];

      double val = 0;
      int th_id = threadIdx.x % 32;
      while (th_id < nTheta) {
        val = val + (DPtr[atom_index + th_id] * YPtr[voxel_index + th_id]);
        th_id = th_id + 32;
      }
      __syncwarp();
#pragma unroll 5
      for (int j = 16; j >= 1; j = j / 2) {
        val += __shfl_down_sync(FULL_MASK, val, j);
      }
      __syncwarp();
      if ((threadIdx.x % 32) == 0) {
        atomicAdd(&wPtr[fiber_index], val * valuesPtr[k]);
      }
      __syncwarp();
    }
  }
  return;
}