project(cufft_callback)
cmake_minimum_required (VERSION 2.8)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/external/cmake_FindCUDA/cmake/Modules" ${CMAKE_MODULE_PATH})

find_package(CUDA)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_BUILD_WITH_CUFFT_STATIC ON)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11" )
include(ExternalProject)
externalproject_add(
  criterion_build
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/criterion"
  INSTALL_COMMAND ""
  )
ExternalProject_Get_Property(criterion_build binary_dir) 
MESSAGE("criterion bin_dir - ${binary_dir}")
link_directories(${binary_dir})
include_directories("${CMAKE_SOURCE_DIR}/external/criterion/include")

include_directories(${PROJECT_SOURCE_DIR})
CUDA_ADD_EXECUTABLE(
  cufft_callback_test.x
  cufft_callback_test.cu
  )

TARGET_LINK_LIBRARIES(
  cufft_callback_test.x
  criterion
  )
