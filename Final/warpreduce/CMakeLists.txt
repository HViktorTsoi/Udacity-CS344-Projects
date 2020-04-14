############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

file(GLOB warpreduce part_a/*.h part_b/*.h)

SET(warpreduce_files_part_a part_a/warpreduce.cu part_a/main.cpp)
SET(warpreduce_files_part_b part_b/warpreduce.cu part_b/main.cpp)

#CUDA_ADD_EXECUTABLE(smooth ${smooth_files} ${smooth})
CUDA_ADD_EXECUTABLE(part_a ${warpreduce_files_part_a} ${warpreduce})
CUDA_ADD_EXECUTABLE(part_b ${warpreduce_files_part_b} ${warpreduce})
