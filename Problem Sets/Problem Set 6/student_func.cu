//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */


#include "utils.h"
#include <thrust/host_vector.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <math_functions.h>

#define FLAG_INTERIOR 128
#define FLAG_BORDER 255
#define FLAG_BLANK 0

// 近邻搜索的四个方向
__device__ const int neighbor[4][2] = {{0,  1},
                                       {0,  -1},
                                       {1,  0},
                                       {-1, 0}};

__global__ void separate_channel(uchar4 const *const d_input_AOS,
                                 const size_t size,
                                 unsigned char *const d_R,
                                 unsigned char *const d_G,
                                 unsigned char *const d_B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    d_R[tid] = d_input_AOS[tid].x;
    d_G[tid] = d_input_AOS[tid].y;
    d_B[tid] = d_input_AOS[tid].z;
//    printf("%d\n", tid);
}

__global__ void calc_mask(unsigned char const *const d_sourceImg_R,
                          unsigned char const *const d_sourceImg_G,
                          unsigned char const *const d_sourceImg_B,
                          const size_t size,
                          unsigned char *const d_mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    if (d_sourceImg_R[tid] == 255 && d_sourceImg_G[tid] == 255 && d_sourceImg_B[tid] == 255) {
        d_mask[tid] = FLAG_BLANK;
    } else {
        // 初始计算mask时所有区域都是内部
        d_mask[tid] = FLAG_INTERIOR;
    }

}

__global__ void calc_interior(unsigned char const *const d_mask,
                              const size_t numRows,
                              const size_t numCols,
                              unsigned char *const d_interior) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x < 0 || tid_y < 0 || tid_x >= numCols || tid_y >= numRows) {
        return;
    }

    // 查询4个邻居
    int num_neighbors = 0;
    for (int i = 0; i < 4; ++i) {
        int x = tid_x + neighbor[i][0], y = tid_y + neighbor[i][1];
        // 保证在mask边界中
        if (x < 0 || y < 0 || x >= numCols || y >= numRows) {
            continue;
        }
        // 查询neighbor是否在mask中为true
        if (d_mask[y * numCols + x] == FLAG_INTERIOR) {
            num_neighbors++;
        }
    }

    if (num_neighbors == 4) {
        // 内部像素
        d_interior[tid_y * numCols + tid_x] = FLAG_INTERIOR;
    } else if (num_neighbors > 0) {
        // 边界像素
        d_interior[tid_y * numCols + tid_x] = FLAG_BORDER;
    } else {
        // 非mask像素
        d_interior[tid_y * numCols + tid_x] = FLAG_BLANK;
    }
}

/*
 * 用source图像初始化求解的buffer
 */
__global__ void init_prev_buffer(unsigned char const *const d_sourceImg,
                                 const size_t size,
                                 float *const d_buffer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    d_buffer[tid] = (float) d_sourceImg[tid];
}

/*
 * 求解一轮
 */
__global__ void solve_one_iteration(unsigned char const *const d_interior,// In
                                    unsigned char const *const d_sourceImg,// In
                                    unsigned char const *const d_destImg,// In
                                    float const *const d_buffer_prev,// In
                                    const size_t numRows,// In
                                    const size_t numCols,// In
                                    float *const d_buffer_next) { // Out
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x < 0 || tid_y < 0 || tid_x >= numCols || tid_y >= numRows) {
        return;
    }
    int tid = tid_y * numCols + tid_x;

    if (d_interior[tid] == FLAG_INTERIOR) {
        // 只计算内部像素
        // 计算4个邻居的sum
        float sum1 = 0, sum2 = 0;
        for (int i = 0; i < 4; ++i) {
            int x = tid_x + neighbor[i][0], y = tid_y + neighbor[i][1];
            // 保证在mask边界中
            if (x < 0 || y < 0 || x >= numCols || y >= numRows) {
                continue;
            }

            int tid_neighbor = y * numCols + x;

            // 1. 计算 sum1
            // If the neighbor is in the interior then += ImageGuess_prev[neighbor]
            // else if the neighbor in on the border then += DestinationImg[neighbor]
            if (d_interior[tid_neighbor] == FLAG_INTERIOR) {
                sum1 += d_buffer_prev[tid_neighbor];
            } else if (d_interior[tid_neighbor] == FLAG_BORDER) {
                sum1 += d_destImg[tid_neighbor];
            }

            // 2. 计算 sum2
            // SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
            sum2 += d_sourceImg[tid] - d_sourceImg[tid_neighbor];
        }

        // 计算最终的像素值
        float new_pixel = (sum1 + sum2) / 4.f;
        d_buffer_next[tid] = fmin(255, fmax(0, new_pixel));

    } else {
        // 跳过外部像素
//        d_buffer_next[tid] = FLAG_BLANK;
        d_buffer_next[tid] = d_destImg[tid];
    }
}


//__global__ void output(unsigned char const *input,
__global__ void output(float const *input,
                       const size_t size,
                       uchar4 *const d_blendedImg,
                       const int ch_id = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    switch (ch_id) {
        case 0:
            d_blendedImg[tid].x = input[tid];
            break;
        case 1:
            d_blendedImg[tid].y = input[tid];
            break;
        case 2:
            d_blendedImg[tid].z = input[tid];
            break;
        default:
            break;
    }

}

void your_blend(const uchar4 *const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4 *const h_destImg, //IN
                uchar4 *const h_blendedImg) //OUT
{

    /* To Recap here are the steps you need to implement

       1) Compute a mask of the pixels from the source image to be copied
          The pixels that shouldn't be copied are completely white, they
          have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

       2) Compute the interior and border regions of the mask.  An interior
          pixel has all 4 neighbors also inside the mask.  A border pixel is
          in the mask itself, but has at least one neighbor that isn't.

       3) Separate out the incoming image into three separate channels

       4) Create two float(!) buffers for each color channel that will
          act as our guesses.  Initialize them to the respective color
          channel of the source image since that will act as our intial guess.

       5) For each color channel perform the Jacobi iteration described
          above 800 times.

       6) Create the output image by replacing all the interior pixels
          in the destination image with the result of the Jacobi iterations.
          Just cast the floating point values to unsigned chars since we have
          already made sure to clamp them to the correct range.

        Since this is final assignment we provide little boilerplate code to
        help you.  Notice that all the input/output pointers are HOST pointers.

        You will have to allocate all of your own GPU memory and perform your own
        memcopies to get data in and out of the GPU memory.

        Remember to wrap all of your calls with checkCudaErrors() to catch any
        thing that might go wrong.  After each kernel call do:

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        to catch any errors that happened while executing the kernel.
    */
    size_t size = numColsSource * numRowsSource;

    // 分配内存
    uchar4 *d_sourceImg, *d_destImg, *d_blendedImg;

    unsigned char *d_mask, *d_interior;
    unsigned char *d_sourceImg_R, *d_sourceImg_G, *d_sourceImg_B;
    unsigned char *d_destImg_R, *d_destImg_G, *d_destImg_B;
    float *d_buffer_prev, *d_buffer_next;

    checkCudaErrors(cudaMalloc(&d_mask, size * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_interior, size * sizeof(unsigned char)));

    checkCudaErrors(cudaMalloc(&d_sourceImg, size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&d_sourceImg_R, size * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_sourceImg_G, size * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_sourceImg_B, size * sizeof(unsigned char)));

    checkCudaErrors(cudaMalloc(&d_destImg, size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&d_destImg_R, size * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_destImg_G, size * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_destImg_B, size * sizeof(unsigned char)));

    checkCudaErrors(cudaMalloc(&d_blendedImg, size * sizeof(uchar4)));

    checkCudaErrors(cudaMalloc(&d_buffer_prev, size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&d_buffer_next, size * sizeof(uchar4)));


    int NUM_THREAD_1D = 1024, NUM_BLOCKS_1D = size / NUM_THREAD_1D + 1, NUM_THREAD_2D = 32;
    int NUM_BLOCKS_X = numColsSource / NUM_THREAD_2D + 1, NUM_BLOCKS_Y = numRowsSource / NUM_THREAD_2D + 1;
    int NUM_ITERATION = 800;

    // 1. 分割通道
    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, size * sizeof(uchar4), cudaMemcpyHostToDevice));
    // source
    separate_channel << < NUM_BLOCKS_1D, NUM_THREAD_1D >> >
                                         (d_sourceImg, size, d_sourceImg_R, d_sourceImg_G, d_sourceImg_B);
    checkCudaErrors(cudaDeviceSynchronize());
    // destination
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, size * sizeof(uchar4), cudaMemcpyHostToDevice));
    separate_channel << < NUM_BLOCKS_1D, NUM_THREAD_1D >> >
                                         (d_destImg, size, d_destImg_R, d_destImg_G, d_destImg_B);
    checkCudaErrors(cudaDeviceSynchronize());


    // 2. 计算mask
    calc_mask << < NUM_BLOCKS_1D, NUM_THREAD_1D >> > (d_sourceImg_R, d_sourceImg_G, d_sourceImg_B, size, d_mask);
    checkCudaErrors(cudaDeviceSynchronize());


    // 3. 计算interior
    calc_interior << < dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y), dim3(NUM_THREAD_2D, NUM_THREAD_2D) >> >
                                                         (d_mask, numRowsSource, numColsSource, d_interior);
    checkCudaErrors(cudaDeviceSynchronize());


    // 4. 迭代求解
    unsigned char *p_ch_source[] = {d_sourceImg_R, d_sourceImg_G, d_sourceImg_B};
    unsigned char *p_ch_dest[] = {d_destImg_R, d_destImg_G, d_destImg_B};

    // 依次求解3个通道
    for (int ch_id = 0; ch_id < 3; ++ch_id) {
        // 初始化buffer
        init_prev_buffer << < NUM_BLOCKS_1D, NUM_THREAD_1D >> > (p_ch_source[ch_id], size, d_buffer_prev);
        checkCudaErrors(cudaDeviceSynchronize());

        for (int iter = 0; iter < NUM_ITERATION; ++iter) {
            std::cout << "ITER: " << iter << std::endl;
            // 迭代1次
            solve_one_iteration << < dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y), dim3(NUM_THREAD_2D, NUM_THREAD_2D) >> >
                                                                       (d_interior, p_ch_source[ch_id], p_ch_dest[ch_id],
                                                                               d_buffer_prev, numRowsSource, numColsSource,
                                                                               d_buffer_next);
            checkCudaErrors(cudaDeviceSynchronize());

            // 用新的buffer替代旧的buffer, 这里直接交换两个buffer的地址
            float *tmp = d_buffer_prev;
            d_buffer_prev = d_buffer_next;
            d_buffer_next = tmp;
        }

        // output
        output << < NUM_BLOCKS_1D, NUM_THREAD_1D >> > (d_buffer_next, size, d_blendedImg, ch_id);
        checkCudaErrors(cudaDeviceSynchronize());
    }



    // debug output
//    output << < NUM_BLOCKS_1D, NUM_THREAD_1D >> > (d_interior, size, d_blendedImg);
////    debug_output << < NUM_BLOCKS_1D, NUM_THREAD_1D >> > (d_mask, size, d_blendedImg);
//    checkCudaErrors(cudaDeviceSynchronize());

    // 5. 将结果写回到host中
    checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, size * sizeof(uchar4), cudaMemcpyDeviceToHost));

    // 释放内存
    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_interior));

    checkCudaErrors(cudaFree(d_sourceImg));
    checkCudaErrors(cudaFree(d_sourceImg_R));
    checkCudaErrors(cudaFree(d_sourceImg_G));
    checkCudaErrors(cudaFree(d_sourceImg_B));

    checkCudaErrors(cudaFree(d_destImg));
    checkCudaErrors(cudaFree(d_destImg_R));
    checkCudaErrors(cudaFree(d_destImg_G));
    checkCudaErrors(cudaFree(d_destImg_B));

    checkCudaErrors(cudaFree(d_blendedImg));

    checkCudaErrors(cudaFree(d_buffer_prev));
    checkCudaErrors(cudaFree(d_buffer_next));

}
