cmake_minimum_required(VERSION 3.12)
project(blockcodec_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# std::thread 需要 pthread（base/patch 任务级并行，默认运行时关闭）
find_package(Threads REQUIRED)

add_library(bc_core STATIC src/blockcodec.cpp)
target_link_libraries(bc_core PUBLIC Threads::Threads)

# 唯一加速：分桶 GEMM（cblas_sgemm）。默认开启，找不到 BLAS 时自动回退到纯 C++。
# 如需手动指定库：cmake -DBLAS_LIB=/path/libopenblas.so
option(BC_USE_BLAS "Link CBLAS for batched gemm (the single acceleration)" ON)
if(BC_USE_BLAS)
  if(DEFINED BLAS_LIB)
    target_compile_definitions(bc_core PUBLIC BC_HAS_BLAS)
    target_link_libraries(bc_core PUBLIC ${BLAS_LIB})
  else()
    find_package(BLAS)
    if(BLAS_FOUND)
      target_compile_definitions(bc_core PUBLIC BC_HAS_BLAS)
      target_link_libraries(bc_core PUBLIC ${BLAS_LIBRARIES})
    endif()
  endif()
endif()

add_executable(bench_blockcodec src/bench_main.cpp)
target_link_libraries(bench_blockcodec PRIVATE bc_core)

add_executable(profile_blockcodec src/profile_main.cpp)
target_link_libraries(profile_blockcodec PRIVATE bc_core)

add_executable(dump_blockcodec src/dump_main.cpp)
target_link_libraries(dump_blockcodec PRIVATE bc_core)
