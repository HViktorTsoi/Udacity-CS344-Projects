############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

file(GLOB batcher *.h)

SET(batcher_files batcher.cu main.cpp)

#CUDA_ADD_EXECUTABLE(smooth ${smooth_files} ${smooth})
CUDA_ADD_EXECUTABLE(batcher ${batcher_files} ${batcher})
