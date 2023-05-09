# BSD 3 Clause
# Copyright 2023 Advanced Micro Devices, Inc.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fmha_api)

add_compile_definitions(
  TORCH_EXTENSION_NAME=libfmha_api 
)

IF(NOT DEFINED ENV{ROCM_PATH})
  SET(ROCM_PATH /opt/rocm)
ELSE()
  SET(ROCM_PATH $ENV{ROCM_PATH})
ENDIF()
if(NOT DEFINED ENV{ROCM_INCLUDE_DIRS})
  set(ROCM_INCLUDE_DIRS ${ROCM_PATH}/include)
else()
  set(ROCM_INCLUDE_DIRS $ENV{ROCM_INCLUDE_DIRS})
endif()
# HIP_PATH
IF(NOT DEFINED ENV{HIP_PATH})
  SET(HIP_PATH ${ROCM_PATH}/hip)
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

IF(NOT EXISTS ${HIP_PATH})
  return()
ENDIF()

# HCC_PATH
IF(NOT DEFINED ENV{HCC_PATH})
  SET(HCC_PATH ${ROCM_PATH}/hcc)
ELSE()
  SET(HCC_PATH $ENV{HCC_PATH})
ENDIF()

# HSA_PATH
IF(NOT DEFINED ENV{HSA_PATH})
  SET(HSA_PATH ${ROCM_PATH}/hsa)
ELSE()
  SET(HSA_PATH $ENV{HSA_PATH})
ENDIF()


# Add HIP to the CMAKE Module Path
set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})

find_package(HIP)

set(CMAKE_CXX_COMPILER /opt/rocm/hip/bin/hipcc)
set(CMAKE_CXX_STANDARD 20)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel)
  execute_process(COMMAND git clone https://github.com/ROCmSoftwarePlatform/composable_kernel.git ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel)
  execute_process(COMMAND git checkout 5736b460d WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel)
endif()

execute_process(
	COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
	OUTPUT_VARIABLE tcoutput
)
string(STRIP "${tcoutput}" tcoutput)
message(STATUS "Torch CMake: ${tcoutput}")

list(APPEND CMAKE_PREFIX_PATH "${tcoutput}")
find_package(Torch REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/library/include)
include_directories(${Python3_INCLUDE_DIRS})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src FLA_SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/library/src/utility CK_SRCS)

add_library(fmha_api SHARED fmha_api.cpp ${FLA_SRCS} ${CK_SRCS})
target_link_libraries(fmha_api "${TORCH_LIBRARIES}" "${Python3_LIBRARIES}" "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")
