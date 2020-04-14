#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <cmath>
#include "compare.h"
#include "gputimer.h"

// http://en.wikipedia.org/wiki/Bitonic_sort
__global__ void batcherBitonicMergesort64(float *d_out, const float *d_in) {
    // you are guaranteed this is called with <<<1, 64, 64*4>>>
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = d_in[tid];
    __syncthreads();

    for (int stage = 0; stage <= 5; stage++) {
        for (int substage = stage; substage >= 0; substage--) {
            // TODO
            unsigned stride = 1 << substage;
            // 符合以下条件的线程负责排序
            if (tid % (stride << 1) < stride) {
                float larger = fmax(sdata[tid], sdata[tid + stride]);
                float smaller = fmin(sdata[tid], sdata[tid + stride]);

                // 判断tid是属于升降序的哪部分
                unsigned order = 1 << (stage + 2);
                if (tid % order < (order >> 1)) {
                    // 升序
                    sdata[tid] = smaller;
                    sdata[tid + stride] = larger;
                } else {
                    // 降序
                    sdata[tid] = larger;
                    sdata[tid + stride] = smaller;
                }
            }
            __syncthreads();
        }
    }

    d_out[tid] = sdata[tid];
}

int compareFloat(const void *a, const void *b) {
    if (*(float *) a < *(float *) b) return -1;
    if (*(float *) a == *(float *) b) return 0;
    if (*(float *) a > *(float *) b) return 1;
    return 0;                     // should never reach this
}

void batcher_wrapper() {
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float h_sorted[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 1]
        h_in[i] = (float) random() / (float) RAND_MAX;
        h_sorted[i] = h_in[i];
    }
    qsort(h_sorted, ARRAY_SIZE, sizeof(float), compareFloat);

    // declare GPU memory pointers
    float *d_in, *d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    GpuTimer timer;
    timer.Start();
    batcherBitonicMergesort64 << < 1, ARRAY_SIZE, ARRAY_SIZE * sizeof(float) >> > (d_out, d_in);
    timer.Stop();

    printf("Your code executed in %g ms\n", timer.Elapsed());

    // copy back the sum from GPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    compare(h_out, h_sorted, ARRAY_SIZE);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

}
