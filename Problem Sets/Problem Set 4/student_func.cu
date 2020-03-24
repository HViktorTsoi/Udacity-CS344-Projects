//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include "timer.h"
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <iostream>
#include <device_launch_parameters.h>


#define MAX_THREAD 1024
/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

/*
 * 计算各个数位上的数字分布
 */
__global__ void histogram(
        unsigned int const *const d_inputVals,
        unsigned *const d_hist,
        const size_t numElems) {
    // 行ID代表数据的位置
    int data_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 列ID代表第几个位
    int bit_tid = blockIdx.y * blockDim.y + threadIdx.y;
    // 在当前block中是第几列
    int local_column_tid = threadIdx.x;
    if (data_tid >= numElems) {
        return;
    }

    // 共享存储local hist
    __shared__ extern unsigned shared_hist[];
    //初始化 每个block中第一列负责初始化即可
    // hist的格式[第0位0个数，第0位1个数,第1位0个数，第1位1个数...第31位1个数]
    if (local_column_tid == 0) {
        shared_hist[bit_tid * 2 + 0] = 0;
        shared_hist[bit_tid * 2 + 1] = 0;
    }
    __syncthreads();

    // mask位置
    const unsigned mask = 1 << bit_tid;
    unsigned this_bit_bin = (d_inputVals[data_tid] & mask) >> bit_tid;
//    printf("%d %d %d  %d M:%d D:%d\n",
//           data_tid, bit_tid, local_column_tid, this_bit_bin, mask, d_inputVals[data_tid]);
    // 统计直方图
    atomicAdd(&(shared_hist[bit_tid * 2 + this_bit_bin]), 1);
    __syncthreads();


    // 写回global mem， 还是第一列线程负责写回即可
    if (local_column_tid == 0) {
        atomicAdd(&(d_hist[bit_tid * 2 + 0]), shared_hist[bit_tid * 2 + 0]);
        atomicAdd(&(d_hist[bit_tid * 2 + 1]), shared_hist[bit_tid * 2 + 1]);
    }
}

/**
 * map的函子，取指定位的切片
 */
struct functor_bit_mask {
    unsigned shift, mask;
    bool bit_sign;

    explicit functor_bit_mask(unsigned shift, bool bit_sign)
            : shift(shift), bit_sign(bit_sign) {
        mask = 1 << shift;
    };

    __host__ __device__

    unsigned operator()(unsigned item) {
        // 同或，bit_sign是true则返回1模式，否则返回0模式
        return !(bit_sign ^ ((item & mask) >> shift));
    }
};

/**
 * 使用scan得到的结果重新排列
 * @return
 */
__global__ void distribute(unsigned *d_input,
                           unsigned *d_inputPos,
                           unsigned shift,
                           unsigned *zero_index,
                           unsigned *one_index,
                           size_t numElems,
                           unsigned *d_output,
                           unsigned *d_outputPos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numElems) {
        return;
    }
    // 按照当前位是1 0来写入到输出数组
    unsigned bit = (d_input[tid] & (1 << shift)) >> shift;
    if (bit == 0) {
        d_output[zero_index[tid]] = d_input[tid]; // 值排序
        d_outputPos[zero_index[tid]] = d_inputPos[tid]; // 图像上的index的排序
    } else {
        d_output[one_index[tid]] = d_input[tid];// 值排序
        d_outputPos[one_index[tid]] = d_inputPos[tid]; // 图像上的index的排序
    };
}

void your_sort(unsigned int *const d_inputVals,
               unsigned int *const d_inputPos,
               unsigned int *const d_outputVals,
               unsigned int *const d_outputPos,
               const size_t numElems) {

//    unsigned h_input[size];
//    checkCudaErrors(cudaMemcpy(d_input, h_input, sizeof(unsigned) * size, cudaMemcpyHostToDevice));
//    unsigned *d_input, *d_input_pos;
//    checkCudaErrors(cudaMallocManaged(
//            (unsigned **) &d_input, sizeof(unsigned) * size
//    ));
//    checkCudaErrors(cudaMallocManaged(
//            (unsigned **) &d_input_pos, sizeof(unsigned) * size
//    ));
//
//    // 初始化测试数据
//    thrust::sequence(d_input, d_input + size, 5);
//    thrust::sequence(d_input_pos, d_input_pos + size, 0, 3);

//    cudaFree(d_input);
//    cudaFree(d_input_pos);

//    // benchmark
//    GpuTimer timer;
//    timer.Start();
//    thrust::sort(d_input, d_input + size);
//    timer.Stop();
//    std::cout << timer.Elapsed() << "ms" << std::endl;
    int size = numElems;
    int NUM_THREAD = 32, NUM_BLOCKS = size / NUM_THREAD + 1;

    // 1. 计算1和0出现次数
    unsigned *d_hist;
    const unsigned num_bits = (sizeof(unsigned) * 8);
    // hist的大小是bit数*(1 0)
    checkCudaErrors(cudaMallocManaged(
            (unsigned **) &d_hist, sizeof(unsigned) * num_bits * 2
    ));
    histogram << < dim3(NUM_BLOCKS), dim3(NUM_THREAD, num_bits), num_bits * 2 >> > (d_inputVals, d_hist, size);
    checkCudaErrors(cudaDeviceSynchronize());

    // 2. 排序
    thrust::device_vector<unsigned> v_output(d_inputVals, d_inputVals + size),
            v_output_pos(d_inputPos, d_inputPos + size);
//    thrust::device_vector<unsigned> v_output(d_input, d_input + size), v_output_pos(d_input_pos, d_input_pos + size);
    thrust::device_vector<unsigned> v_input(size), v_input_pos(size);
    thrust::device_vector<unsigned> zero_slice(size), one_slice(size);
    for (int bit = 0; bit < num_bits; ++bit) {
        GpuTimer timer;
        timer.Start();
        // v_input = v_output 这里用memcopy效率更高
        checkCudaErrors(cudaMemcpy(
                (unsigned *) thrust::raw_pointer_cast(&v_input[0]),
                (unsigned *) thrust::raw_pointer_cast(&v_output[0]),
                v_input.size() * sizeof(unsigned), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(
                (unsigned *) thrust::raw_pointer_cast(&v_input_pos[0]),
                (unsigned *) thrust::raw_pointer_cast(&v_output_pos[0]),
                v_input.size() * sizeof(unsigned), cudaMemcpyDeviceToDevice));

        // 3. 使用thrust来处理scan
        // 切片出当前位 并用scan来计算索引位置
        // 升序 0在前
        thrust::transform(v_input.begin(), v_input.end(), zero_slice.begin(), functor_bit_mask(bit, false));
        thrust::exclusive_scan(zero_slice.begin(), zero_slice.end(), zero_slice.begin());

        // 在1的地址上要加上前边0的个数大小的偏移
        thrust::transform(v_input.begin(), v_input.end(), one_slice.begin(), functor_bit_mask(bit, true));
        thrust::exclusive_scan(one_slice.begin(), one_slice.end(), one_slice.begin(), d_hist[bit * 2]);

        // 4. 排列数组
        // 这里的zero和one slice就相当于numpy中的indices
        NUM_THREAD = 512, NUM_BLOCKS = size / NUM_THREAD + 1;
        distribute << < NUM_BLOCKS, NUM_THREAD >> > (
                (unsigned *) thrust::raw_pointer_cast(&v_input[0]),
                        (unsigned *) thrust::raw_pointer_cast(&v_input_pos[0]),
                        bit,
                        (unsigned *) thrust::raw_pointer_cast(&zero_slice[0]),
                        (unsigned *) thrust::raw_pointer_cast(&one_slice[0]),
                        size,
                        (unsigned *) thrust::raw_pointer_cast(&v_output[0]),
                        (unsigned *) thrust::raw_pointer_cast(&v_output_pos[0]));
        checkCudaErrors(cudaDeviceSynchronize());

//        std::cout << std::endl;
//        std::cout << "ERR: " << cudaGetLastError() << std::endl;
//
//        for (int i = 0; i < 32; ++i) {
//            std::cout
//                    << zero_slice[i] << " "
//                    << one_slice[i] << "  "
//                    << v_output[i] << " "
//                    << v_output_pos[i] << " \n";
//        }
//        std::cout << std::endl << std::endl;
//        timer.Stop();
//        std::cout << timer.Elapsed() << "ms" << std::endl;
    }

    // 5. 写回结果数组
    checkCudaErrors(cudaMemcpy(d_outputVals, (unsigned *) thrust::raw_pointer_cast(&v_output[0]),
                               size * sizeof(unsigned), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, (unsigned *) thrust::raw_pointer_cast(&v_output_pos[0]),
                               size * sizeof(unsigned), cudaMemcpyDeviceToDevice));

    cudaFree(d_hist);
}
