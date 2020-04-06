/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include "utils.h"

const unsigned S = 32;
texture<unsigned, 1, cudaReadModeElementType> texture_val;
/*
 * 最简单版本的histo
 * 3ms左右
 */
__global__
void histo_naive(const unsigned int *const vals, //INPUT
                 unsigned int *const histo,      //OUPUT
                 int numBins,
                 int numElems) {
    //TODO fill in this kernel to calculate the histogram
    //as quickly as possible

    //Although we provide only one kernel skeleton,
    //feel free to use more if it will help you
    //write faster code
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numElems) {
        return;
    }
    atomicAdd(&(histo[vals[tid]]), 1);
}

/*
 * 使用shared mem的histo 仅能处理bin小于等于blockDim的情况
 * 0.4ms左右
 */
__global__
void histo_shared(const unsigned int *const vals, //INPUT
                  unsigned int *const histo,      //OUPUT
                  int numBins,
                  int numElems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numElems) {
        return;
    }

    // shared mem初始化
    int local_id = threadIdx.x;
    __shared__ extern unsigned shared_hist[];
    if (local_id < numBins) {
        shared_hist[local_id] = 0;
    }
    __syncthreads();

    atomicAdd(&(shared_hist[vals[tid]]), 1);
    __syncthreads();

    //　写回global mem
    if (local_id < numBins) {
        atomicAdd(&(histo[local_id]), shared_hist[local_id]);
    }
}

/*
 * 使用shared mem tiling的histo 可以处理任意大小的hist
 * 0.58ms左右
 */
__global__
void histo_shared_tiling(const unsigned int *const vals, //INPUT
                         unsigned int *const histo,      //OUPUT
                         int numBins,
                         int numElems,
                         int tiling_times) {
    __shared__ extern unsigned shared_hist[];
    int tid = blockIdx.x * (blockDim.x) + threadIdx.x;
    if (tid >= numElems) {
        return;
    }

    // shared mem初始化
    for (int i = 0, local_id = threadIdx.x; i < tiling_times && local_id < numBins; ++i, local_id += blockDim.x) {
        shared_hist[local_id] = 0;
    }
    __syncthreads();

    atomicAdd(&(shared_hist[vals[tid]]), 1);
    __syncthreads();

    //　写回global mem
    for (int i = 0, local_id = threadIdx.x; i < tiling_times && local_id < numBins; ++i, local_id += blockDim.x) {
        atomicAdd(&(histo[local_id]), shared_hist[local_id]);
    }
}

/*
 * 使用shared mem tiling的histo 并且每个线程处理多个bytes的数据(而不是每个线程负责一个数据)
 * 0.2ms左右 开启release之后降到0.076s
 * 使用texture memory之后效果不好 这可能是因为本身也只读取了一次内存
 */
__global__
void histo_shared_tiling_multibytes(const unsigned int *const vals, //INPUT
                                    unsigned int *const histo,      //OUPUT
                                    int numBins,
                                    int numElems,
                                    int tiling_times) {
    __shared__ extern unsigned shared_hist[];

    // shared mem初始化
    for (int i = 0, local_id = threadIdx.x; i < tiling_times && local_id < numBins; ++i, local_id += blockDim.x) {
        shared_hist[local_id] = 0;
    }
    __syncthreads();

    for (int tid = blockIdx.x * (blockDim.x * S) + threadIdx.x, i = 0;
         i < S && tid < numElems; ++i, tid += blockDim.x) {
        // global memory
        atomicAdd(&(shared_hist[vals[tid]]), 1);
        // texture memory
//        atomicAdd(&(shared_hist[tex1Dfetch(texture_val, tid)]), 1);
    }
    __syncthreads();

    //　写回global mem
    for (int i = 0, local_id = threadIdx.x; i < tiling_times && local_id < numBins; ++i, local_id += blockDim.x) {
        atomicAdd(&(histo[local_id]), shared_hist[local_id]);
    }
}

void computeHistogram(const unsigned int *const d_vals, //INPUT
                      unsigned int *const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems) {
    //TODO Launch the yourHisto kernel

    //if you want to use/launch more than one kernel,
    //feel free
//    const int NUM_THREAD = 1024, NUM_BLOCK = numElems / NUM_THREAD + 1;
//    histo_naive << < NUM_BLOCK, NUM_THREAD >> > (d_vals, d_histo, numBins, numElems);
//    histo_shared << < NUM_BLOCK, NUM_THREAD, numBins * sizeof(unsigned) >> > (d_vals, d_histo, numBins, numElems);
//    histo_shared_tiling << < NUM_BLOCK, NUM_THREAD, numBins * sizeof(unsigned) >> >
//                                                    (d_vals, d_histo, numBins, numElems, numBins / NUM_THREAD + 1);
//    const int NUM_THREAD = 1024, NUM_BLOCK = numElems / NUM_THREAD / S + 1;
//    histo_shared_tiling_multibytes << < NUM_BLOCK, NUM_THREAD, numBins * sizeof(unsigned) >> >
//                                                               (d_vals, d_histo, numBins, numElems,
//                                                                       numBins / NUM_THREAD + 1);
    // texture memory
    cudaBindTexture(0, texture_val, d_vals, sizeof(unsigned) * numElems);
    const int NUM_THREAD = 1024, NUM_BLOCK = numElems / NUM_THREAD / S + 1;
    histo_shared_tiling_multibytes << < NUM_BLOCK, NUM_THREAD, numBins * sizeof(unsigned) >> >
                                                               (d_vals, d_histo, numBins, numElems,
                                                                       numBins / NUM_THREAD + 1);
    cudaDeviceSynchronize();
//    std::cout << "ERR: " << cudaGetLastError() << std::endl;
    checkCudaErrors(cudaGetLastError());
    cudaUnbindTexture(texture_val);
}
