#if(BUILD_ANDROID_PACKAGE)
  #ocv_module_disable(nonfree)
#endif()

#set(the_description "Functionality with possible limitations on the use")
#ocv_add_module(nonfree opencv_imgproc opencv_features2d opencv_calib3d OPTIONAL opencv_gpu opencv_ocl)
#ocv_module_include_directories()

#if(HAVE_CUDA AND HAVE_opencv_gpu)
  #ocv_source_group("Src\\Cuda" GLOB "src/cuda/*.cu")
  #ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/include" ${CUDA_INCLUDE_DIRS})
  #ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)

  #file(GLOB lib_cuda "src/cuda/*.cu")
  #ocv_cuda_compile(cuda_objs ${lib_cuda})

  #set(cuda_link_libs ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
#else()
  #set(lib_cuda "")
  #set(cuda_objs "")
  #set(cuda_link_libs "")
#endif()

#ocv_glob_module_sources(SOURCES ${lib_cuda} ${cuda_objs})

#ocv_create_module(${cuda_link_libs})
#ocv_add_precompiled_headers(${the_module})

#ocv_add_accuracy_tests()
#ocv_add_perf_tests()
cmake_minimum_required (VERSION 2.8.11)
project (vibe_gpu)

#set(OBJECT_DETECTION_SRZN_MAJOR_VERSION 1)
#set(OBJECT_DETECTION_SRZN_MINOR_VERSION 0)
#set(OBJECT_DETECTION_SRZN_PATCH_VERSION 4)
#set(OBJECT_DETECTION_SRZN_VERSION
  #${OBJECT_DETECTION_SRZN_MAJOR_VERSION}.${OBJECT_DETECTION_SRZN_MINOR_VERSION}.${OBJECT_DETECTION_SRZN_PATCH_VERSION})

#if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 
  #set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE) 
#endif() 

  #"${PROJECT_BINARY_DIR}")  # to find foo/config.h

#configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

#OPTION(USE_GPU "Use GPU" ON) # Enabled by default
#IF(USE_GPU)
  #ADD_DEFINITIONS(-DGPU=1)
#ENDIF(USE_GPU)

find_package(OpenCV)
include_directories("include;${OpenCV_INCLUDE_DIRETORIES}")

find_package(CUDA)
SET(CUDA_NVCC_FLAGS "--gpu-architecture=compute_20 --gpu-code=compute_20" CACHE STRING "nvcc flags" FORCE)
#CFLAGS+= -DGPU
#cuda_add_library(vibe_gpu SHARED ../src/convolutional_kernels.cu ../src/deconvolutional_kernels.cu ../src/activation_kernels.cu ../src/im2col_kernels.cu ../src/col2im_kernels.cu ../src/blas_kernels.cu ../src/crop_layer_kernels.cu ../src/dropout_layer_kernels.cu ../src/maxpool_layer_kernels.cu ../src/softmax_layer_kernels.cu ../src/network_kernels.cu ../src/avgpool_layer_kernels.cu ../src/gemm.c ../src/utils.c ../src/cuda.c ../src/deconvolutional_layer.c ../src/convolutional_layer.c ../src/list.c ../src/image.c ../src/activations.c ../src/im2col.c ../src/col2im.c ../src/blas.c ../src/crop_layer.c ../src/dropout_layer.c ../src/maxpool_layer.c ../src/softmax_layer.c ../src/matrix.c ../src/network.c ../src/connected_layer.c ../src/cost_layer.c ../src/parser.c ../src/detection_layer.c ../src/route_layer.c ../src/box.c ../src/normalization_layer.c ../src/avgpool_layer.c ../src/layer.c ../src/local_layer.c ../src/shortcut_layer.c ../src/activation_layer.c ../src/rnn_layer.c ../src/option_list.c ../src/data.c object_detection.cpp object_detection.hpp config.h.in)
cuda_add_executable(vibe_gpu src/vibe_gpu.cpp src/vibe.cu src/test_vibe.cpp)
target_link_libraries(vibe_gpu ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${OpenCV_LIBRARIES})

#message(STATUS ${CUDA_LIBRARIES})
# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
#target_include_directories (object_detection_srzn PRIVATE ../src ${CUDA_INCLUDE_DIRS})
##CUDA_ADD_CUBLAS_TO_TARGET(object_detection_srzn)
##target_link_libraries(object_detection_srzn m pthread object_detection_srzn_cuda)
#target_link_libraries(object_detection_srzn m pthread ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})

#set_target_properties(object_detection_srzn PROPERTIES
  #PUBLIC_HEADER "object_detection.hpp;${CMAKE_CURRENT_BINARY_DIR}/config.h"
  #VERSION ${OBJECT_DETECTION_SRZN_VERSION})

##add_subdirectory(test)
#install(TARGETS object_detection_srzn
  ## IMPORTANT: Add the foo library to the "export-set"
  #RUNTIME DESTINATION "bin" COMPONENT bin
  #LIBRARY DESTINATION "lib" COMPONENT shlib
  #PUBLIC_HEADER DESTINATION "include"
  #COMPONENT dev)

#add_subdirectory(test)
##find_package(OpenCV REQUIRED)
##include_directories(${OpenCV_INCLUDE_DIRS})
##add_executable(test test/test.cpp)
##target_link_libraries(test object_detection_srzn ${OpenCV_LIBRARIES})


#install(FILES test/test.cpp
  #test/CMakeLists.txt
  #DESTINATION test)
