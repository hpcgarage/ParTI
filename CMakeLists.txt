cmake_minimum_required(VERSION 3.2)  # CMake 3.2 supports CUDA 7.0
project(ParTI)

option(BUILD_STATIC "Build static library" ON)
option(BUILD_SHARED "Build shared library" ON)

option(USE_CUDA "Use NVIDIA CUDA library" ON)
option(USE_OPENBLAS "Use OpenBLAS library" OFF)
option(USE_LAPACKE "Use LAPACKE library" OFF)
option(USE_MAGMA "Use MAGMA library" OFF)
option(USE_MKL "Use Intel MKL library" OFF)

option(CUDA_HOST_COMPILER "CUDA host compiler (gcc < 6.0)" /usr/bin/gcc-5)

set(Backward_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/backward-cpp")
add_subdirectory("${Backward_DIR}")

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11" "-D_BITS_FLOATN_H")
    add_definitions(-DPARTI_USE_CUDA)
    link_libraries(${CUDA_cusparse_LIBRARY})
    link_libraries(${CUDA_cusolver_LIBRARY})
endif()

# A bug in FindCUDA module forces us to put "link_libraries" before "cuda_add_library".
# So we cannot use "target_link_libraries" for target-wise library tracking.
if(USE_OPENBLAS)
    find_package(OpenBLAS REQUIRED)
    add_definitions(-DPARTI_USE_OPENBLAS)
    add_definitions(-DPARTI_USE_CBLAS)
    add_definitions(-DPARTI_USE_LAPACKE)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    link_libraries(${OpenBLAS_LIBRARIES})
endif()
if(USE_CBLAS AND NOT USE_OPENBLAS)
    find_library(
        CBLAS_LIBRARIES
        NAMES cblas
        PATHS ${LAPACKE_DIR}
    )
    add_definitions(-DPARTI_USE_CBLAS)
    get_filename_component(CBLAS_LIBDIR "${CBLAS_LIBRARIES}" DIRECTORY)
    include_directories("${CBLAS_LIBDIR}/../include")
    link_libraries(${CBLAS_LIBRARIES})
endif()
if(USE_LAPACKE AND NOT USE_OPENBLAS)
    find_library(
        LAPACKE_LIBRARIES
        NAMES lapacke
        PATHS ${LAPACKE_DIR}
    )
    add_definitions(-DPARTI_USE_LAPACKE)
    get_filename_component(LAPACKE_LIBDIR "${LAPACKE_LIBRARIES}" DIRECTORY)
    include_directories("${LAPACKE_LIBDIR}/../include")
    link_libraries(${LAPACKE_LIBRARIES})
endif()
if(USE_MKL)
    find_package(MKL REQUIRED)
    add_definitions(-DPARTI_USE_MKL)
    include_directories(${MKL_INCLUDE_DIRS})
    link_libraries(${MKL_LIBRARIES})
endif()

if(NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHsc /W2")
endif()
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W2")
endif()

# A bug in FindCUDA module forces us to put "include_directories" before "cuda_add_library".
# So we cannot use "target_include_directories" for target-wise include tracking.
include_directories("include")
if(NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    link_libraries("m")
endif()

if(USE_CUDA)
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp" "src/*.cu" "src/*.hpp" "include/*.hpp")
    if(BUILD_SHARED)
        cuda_add_library(ParTI SHARED ${PARTI_SRC})
        CUDA_ADD_CUBLAS_TO_TARGET(ParTI)
    endif()
    if(BUILD_STATIC)
        cuda_add_library(ParTI_s STATIC ${PARTI_SRC})
        CUDA_ADD_CUBLAS_TO_TARGET(ParTI_s)
    endif()
else()
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp" "src/*.hpp" "include/*.hpp")
    if(BUILD_SHARED)
        add_library(ParTI SHARED ${PARTI_SRC})
    endif()
    if(BUILD_STATIC)
        add_library(ParTI_s STATIC ${PARTI_SRC})
    endif()
endif()

if(BUILD_SHARED)
    set_target_properties(ParTI PROPERTIES C_STANDARD 99)
    set_target_properties(ParTI PROPERTIES CXX_STANDARD 11)
endif()
if(BUILD_STATIC)
    set_target_properties(ParTI_s PROPERTIES C_STANDARD 99)
    set_target_properties(ParTI_s PROPERTIES CXX_STANDARD 11)
endif()

#add_subdirectory("examples")
add_subdirectory("tests")
