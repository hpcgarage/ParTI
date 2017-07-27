cmake_minimum_required(VERSION 3.2)  # CMake 3.2 supports CUDA 7.0
project(ParTI)

option(BUILD_STATIC "Build static library" ON)
option(BUILD_SHARED "Build shared library" ON)

option(USE_CUDA "Use NVIDIA CUDA library" ON)
option(USE_OpenBLAS "Use OpenBLAS library" ON)
option(USE_MAGMA "Use MAGMA library" ON)
option(USE_MKL "Use Intel MKL library" OFF)

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DPARTI_USE_CUDA)
    link_libraries(${CUDA_cusparse_LIBRARY})
    link_libraries(${CUDA_cusolver_LIBRARY})
endif()

# A bug in FindCUDA module forces us to put "link_libraries" before "cuda_add_library".
# So we cannot use "target_link_libraries" for target-wise library tracking.
if(USE_OpenBLAS)
    find_package(OpenBLAS REQUIRED)
    add_definitions(-DPARTI_USE_OpenBLAS)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    link_libraries(${OpenBLAS_LIBRARIES})
endif()
if(USE_MAGMA)
    find_library(
        MAGMA_LIBRARIES
        NAMES magma
        PATHS ${MAGMA_DIR}
        NO_DEFAULT_PATH
    )
    add_definitions(-DPARTI_USE_MAGMA)
    get_filename_component(MAGMA_LIBDIR "${MAGMA_LIBRARIES}" DIRECTORY)
    include_directories("${MAGMA_LIBDIR}/../include")
    link_libraries(${MAGMA_LIBRARIES})
endif()
if(USE_MKL)
    find_package(MKL REQUIRED)
    add_definitions(-DPARTI_USE_MKL)
    include_directories(${MKL_INCLUDE_DIRS})
    link_libraries(${MKL_LIBRARIES})
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp -O3")

# A bug in FindCUDA module forces us to put "include_directories" before "cuda_add_library".
# So we cannot use "target_include_directories" for target-wise include tracking.
include_directories("include")
link_libraries("m")

if(USE_CUDA)
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.cu" "src/*.h" "include/*.h")
    # list(REMOVE_ITEM PARTI_SRC "src/sptensor/all_splits.c" "src/sptensor/cpd*.c" "src/sptensor/cpd*.cu" "src/sptensor/finesplit.c" "src/sptensor/mediumsplit.c" "src/sptensor/coarsesplit.c" "src/sptensor/mttkrp_cuda_coarse.cu" "src/sptensor/mttkrp_cuda_dist.cu" "src/sptensor/mttkrp_split.c")
    if(BUILD_SHARED)
        cuda_add_library(ParTI SHARED ${PARTI_SRC})
        CUDA_ADD_CUBLAS_TO_TARGET(ParTI)
    endif()
    if(BUILD_STATIC)
        cuda_add_library(ParTI_s STATIC ${PARTI_SRC})
        CUDA_ADD_CUBLAS_TO_TARGET(ParTI_s)
    endif()
else()
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.h" "include/*.h")
    if(BUILD_SHARED)
        add_library(ParTI SHARED ${PARTI_SRC})
    endif()
    if(BUILD_STATIC)
        add_library(ParTI_s STATIC ${PARTI_SRC})
    endif()
endif()

if(BUILD_SHARED)
    set_target_properties(ParTI PROPERTIES C_STANDARD 99)
endif()
if(BUILD_STATIC)
    set_target_properties(ParTI_s PROPERTIES C_STANDARD 99)
endif()

add_subdirectory("examples")
add_subdirectory("tests")
