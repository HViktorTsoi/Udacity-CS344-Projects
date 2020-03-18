/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <device_launch_parameters.h>
#include <math_functions.h>
#include <cstdio>
#include "utils.h"
#include "timer.h"

#define MAX_THREAD 1024

typedef float(*reduce_function_t)(float, float);

__device__ reduce_function_t d_func_fminf = fminf;
__device__ reduce_function_t d_func_fmaxf = fmaxf;

/*
 * Do reduce
 */
__global__ void reduce_kernel(const float *const d_in,
                              float *d_out,
                              const size_t size,
                              reduce_function_t reduce_func) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_local_tid = threadIdx.x;
    // boundary
    if (tid >= size) {
        return;
    }

    // 存储到共享内存中
    // TODO 处理block中实际元素不满blockDim的情况，注意此处不能直接返回，因为这一步只是拷贝数据，后边的处理中，tid-stride编号
    //  的线程也会依赖这个线程的数据，所以只能将shared mem补齐
    __shared__ extern float shared_mem[];
    shared_mem[block_local_tid] = d_in[tid];
    __syncthreads();

    // 找到blocks中的最小值
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // 处理一半的数据 保证没有冲突
        // 这里也可以用i, i+1的模式， 条件改为block_local_tid%2==0就可以，类似的如“哲学家就餐问题”
        if (block_local_tid < stride) {
            // 处理边角余料，用序列里第一个元素取代替或者可以叫pad多余的位置
            int adj_idx = tid + stride < size ? block_local_tid + stride : 0;
            shared_mem[block_local_tid] = (*reduce_func)(shared_mem[block_local_tid], shared_mem[adj_idx]);
//            printf("%d %d\n", tid, tid + stride);
        }
        // barrier， 保证所有线程执行完毕
        __syncthreads();
    }

    // 写回，此时所有的blocks里都只有一个元素了，该元素对应的线程负责写回
    if (block_local_tid == 0) {
        d_out[blockIdx.x] = shared_mem[block_local_tid];
//        printf("%d\n", tid);
    }
}

/*
 * reduce wrapper
 */
float reduce_kernel_wrapper(reduce_function_t op, const float *const d_in, const size_t size, const int num_thread) {
    float *d_inter;
    float result;
    checkCudaErrors(cudaMalloc(&d_inter, size * sizeof(float)));

    // 多趟reduce，每一次都将数量reduce到size/num_threads(或称之为num_blocks)
    for (size_t cur_size = size; cur_size > 0; cur_size /= num_thread) {
        // block数量是元素个数/thread数量
        int num_blocks = cur_size / num_thread + int(cur_size % num_thread > 0);
        // 调用内核
        reduce_kernel << < num_blocks, num_thread, num_thread * sizeof(float) >> > (
                cur_size == size ? d_in : d_inter,// 第一次输入是原始数据，后边的输入都是中间数据
                        d_inter,
                        cur_size,
                        op
        );
    }
    // copy first element as result to host
    checkCudaErrors(cudaMemcpy(&result, &(d_inter[0]), sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_inter);
    return result;
}

/*
 * find the min and max
 */
float3 find_min_max(const float *const d_logLuminance, const size_t size) {
    // intermidate memory
    reduce_function_t h_reduce_function;
    float min_val, max_val;
    const int NUM_THREAD = 1024;

    // find min
    checkCudaErrors(cudaMemcpyFromSymbol(&h_reduce_function, d_func_fminf, sizeof(reduce_function_t)));
    min_val = reduce_kernel_wrapper(h_reduce_function, d_logLuminance, size, NUM_THREAD);

    // find max
    checkCudaErrors(cudaMemcpyFromSymbol(&h_reduce_function, d_func_fmaxf, sizeof(reduce_function_t)));
    max_val = reduce_kernel_wrapper(h_reduce_function, d_logLuminance, size, NUM_THREAD);

    // calc range
    float range = max_val - min_val;

    // return result
    float3 result;
    result.x = min_val, result.y = max_val, result.z = range;
    return result;
}


/*
 * histogram
 */

__global__ void histogram_kernel(const float *const d_in,
                                 unsigned int *const d_hist,
                                 const float lumRange,
                                 const float lumMin,
                                 const float lumMax,
                                 const size_t numBins,
                                 const size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_local_id = threadIdx.x;
    __shared__ extern unsigned int shared_hist[];

//    printf("%f %f %f %f %ld %ld\n", d_in[tid], lumMin, lumMax, lumRange, numBins, size);

    // boundary
    if (tid >= size) {
        return;
    }

    // 复制到shared mem, 并初始化
    if (block_local_id < numBins) {
        shared_hist[block_local_id] = 0;
    }
    __syncthreads();

    // 取numBins - 1和实际bin的最小值，防止到最大值时溢出
    int bin = min(numBins - 1, (unsigned long) ((d_in[tid] - lumMin) / lumRange * numBins));
//    printf("%d\n", bin);
    atomicAdd(&(shared_hist[bin]), 1);
    __syncthreads();

    if (block_local_id < numBins) {
        // 将shared mem中的hist加回global memory
        atomicAdd(&(d_hist[block_local_id]), shared_hist[block_local_id]);
    }

}

/*
 * scan for per block
 */
__global__ void blelloch_scan_kernel(unsigned int *const d_cdf,
                                     unsigned int *const d_tmp_sum,
                                     const size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_local_id = threadIdx.x;
    __shared__ extern unsigned int shared_cdf[];
    // 拷贝到shared mem
    // TODO 这里对于sum运算 单位元就是0 可能有其他运算如min max等需要其他单位元
    shared_cdf[block_local_id] = tid < size ? d_cdf[tid] : 0;

    // TODO 保证blockDim是2的整数次方
    // 首先自顶向下做inplace的reduce
    for (size_t stride = 2; stride <= blockDim.x; stride <<= 1) {
        __syncthreads();
        // 如果当前tid为负责计算的tid 则d[block_local_id]=d[block_local_id] op d[block_local_id-stride/2]
        if (block_local_id % stride == stride - 1) {
//            d_cdf[block_local_id] = d_cdf[block_local_id] + d_cdf[block_local_id - (stride >> 1)];
            shared_cdf[block_local_id] = shared_cdf[block_local_id] + shared_cdf[block_local_id - (stride >> 1)];
        }
    }
    // 最后一个数字置0
    if (block_local_id == blockDim.x - 1) {
        shared_cdf[block_local_id] = 0;
    }
    __syncthreads();

    // downswap
    for (size_t stride = blockDim.x; stride > 1; stride >>= 1) {
        if (block_local_id % stride == stride - 1) {
            unsigned int _copy = shared_cdf[block_local_id];
            shared_cdf[block_local_id] = shared_cdf[block_local_id] + shared_cdf[block_local_id - (stride >> 1)];
            shared_cdf[block_local_id - (stride >> 1)] = _copy;
        }
        __syncthreads();
    }

    // 保存当前block的总和 就是结果数组的最后一个元素加上输入的最后一个元素
    if (d_tmp_sum != NULL && block_local_id == blockDim.x - 1) {
        d_tmp_sum[blockIdx.x] = d_cdf[tid] + shared_cdf[block_local_id];
    }

    // 写回global mem
    if (tid < size) {
        d_cdf[tid] = shared_cdf[block_local_id];
    }

}

/*
 * 将各个block中累积和与scan的原始结果相加
 */
__global__ void broadcast_add_kernel(unsigned int *const d_cdf,
                                     unsigned int *const d_tmp_sum,
                                     const size_t raw_block_dim,
                                     const size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    // 把对应块的sum加到原始数据
    d_cdf[tid] += d_tmp_sum[tid / raw_block_dim];
}

void integral(unsigned int *const d_hist,
              unsigned int *const d_cdf,
              size_t size) {

    int NUM_THREAD = 1024, NUM_BLOCK = size / NUM_THREAD + int(size % NUM_THREAD > 0);
    unsigned int *d_tmp_sum;
    checkCudaErrors(cudaMalloc(&d_tmp_sum, NUM_BLOCK * sizeof(unsigned int)));

    // 这里保证所有block得到的原始结果的累积和数量小于MAX_THREAD 这样只要再对累积和原始数组进行一次scan(大小为一个block)就能得到原始数组
    assert(size < NUM_THREAD * MAX_THREAD);

    // 首先对每个block计算scan结果 并原地存储
    blelloch_scan_kernel << < NUM_BLOCK, NUM_THREAD, NUM_THREAD * sizeof(unsigned int) >> > (d_hist, d_tmp_sum, size);

    // 计算每个block的sum的cumsum 这里假定第一步得到的sum数量小于MAX_THREAD 这样在一个block内可以处理完成
    blelloch_scan_kernel << < 1, MAX_THREAD, MAX_THREAD * sizeof(unsigned int) >> > (d_tmp_sum, NULL, NUM_BLOCK);

    // 将累积和加到原始结果上
    broadcast_add_kernel << < NUM_BLOCK, NUM_THREAD >> > (d_hist, d_tmp_sum, NUM_THREAD, size);

    // 复制结果到输出
    checkCudaErrors(cudaMemcpy(d_cdf, d_hist, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    cudaFree(d_tmp_sum);
}

void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins) {
    //TODO
    /*Here are the steps you need to implement
      1) find the minimum and maximum value in the input logLuminance channel
         store in min_logLum and max_logLum
      2) subtract them to find the range
      3) generate a histogram of all the values in the logLuminance channel using
         the formula: bin = (lum[i] - lumMin) / lumRange * numBins
      4) Perform an exclusive scan (prefix sum) on the histogram to get
         the cumulative distribution of luminance values (this should go in the
         incoming d_cdf pointer which already has been allocated for you)       */


//    // TEST
//    const int SIZE = 100;
//    const int NUM_THREAD = 16, NUM_BLOCK = SIZE / NUM_THREAD + int(SIZE % NUM_THREAD > 0);
//    unsigned int h_in[SIZE], h_out[NUM_BLOCK], h_tmp_sum[NUM_BLOCK];
//    unsigned int h_hist[numBins];
//    for (int i = 0; i < SIZE; ++i) {
//        h_in[i] = i + 1;
//    }
//
//    unsigned int *d_hist;
//    checkCudaErrors(cudaMalloc(&d_hist, numBins * sizeof(unsigned int)));
    //    unsigned int *d_in, *d_out;
//    unsigned int *d_tmp_sum;
//    checkCudaErrors(cudaMalloc(&d_in, SIZE * sizeof(unsigned int)));
//    checkCudaErrors(cudaMalloc(&d_tmp_sum, NUM_BLOCK * sizeof(unsigned int)));
//    checkCudaErrors(cudaMalloc(&d_out, NUM_BLOCK * sizeof(unsigned int)));
//    checkCudaErrors(cudaMemcpy(d_in, h_in, SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice));
//
//    GpuTimer timer;
//    timer.Start();
//
//    integral(d_in, d_in, SIZE);
//    checkCudaErrors(cudaDeviceSynchronize());
//    checkCudaErrors(cudaMemcpy(h_in, d_in, SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//
//    timer.Stop();
//    printf("\nERR: %d\n", cudaGetLastError());
//    printf("Your code ran in: %f msecs.\n", timer.Elapsed());
//    for (int i = 0; i < SIZE; ++i) {
//        if (i % 20 == 0) {
//            printf("\n");
//        }
//        printf("%d, ", h_in[i]);
//    }
//    printf("\n\n");
//
//    checkCudaErrors(cudaMemcpy(h_hist, d_hist, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//    for (int i = 0; i < numBins; ++i) {
//        if (i % 20 == 0) {
//            printf("\n");
//        }
//        printf("%d, ", h_hist[i]);
//    }
//
//    cudaFree(d_in);
//    cudaFree(d_out);
//    cudaFree(d_hist);
//
//    GpuTimer timer;
//    timer.Start();

    const int size = numRows * numCols;
    unsigned int *d_hist;
    checkCudaErrors(cudaMalloc(&d_hist, numBins * sizeof(unsigned int)));

    // 1. 找到最大最小值
    float3 val = find_min_max(d_logLuminance, size);
    min_logLum = val.x;
    max_logLum = val.y;
    checkCudaErrors(cudaDeviceSynchronize());
    printf("size:%ld, %f, %f, %f", numRows * numCols, val.x, val.y, val.z);
    printf("\n\n");

    // 2. 求统计直方图
    // TODO 为了支持任意大小的numBins, 需要将kernel改成N次循环处理，每次tid跳过N个block，
    // TODO 把N个block对应的统计结果汇总到同一个直方图中，参考CUDA Book的P102
    const int NUM_THREAD = 1024, NUM_BLOCK = size / NUM_THREAD + int(size % NUM_THREAD > 0);
    assert(numBins <= NUM_THREAD);
    histogram_kernel << < NUM_BLOCK, NUM_THREAD, numBins * sizeof(unsigned int) >> > (
            d_logLuminance, d_hist, val.z, min_logLum, max_logLum, numBins, numRows * numCols);
    checkCudaErrors(cudaDeviceSynchronize());

    // 3. 对直方图进行scan， 求cumsum
    integral(d_hist, d_cdf, numBins);
    checkCudaErrors(cudaDeviceSynchronize());
//
//    timer.Stop();
//    unsigned int h_hist[numBins];
//    checkCudaErrors(cudaMemcpy(h_hist, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//    for (int i = 0; i < numBins; ++i) {
//        if (i % 20 == 0) {
//            printf("\n");
//        }
//        printf("%d, ", h_hist[i]);
//    }
//
//    printf("\nERR: %d\n", cudaGetLastError());
//    printf("Your code ran in: %f msecs.\n", timer.Elapsed());

    cudaFree(d_hist);

}
