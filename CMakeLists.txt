cmake_minimum_required(VERSION 3.10)
project(csmg LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE on)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE RELEASE)

include_directories(${PROJECT_SOURCE_DIR})

# make sure cuda_version >= 11.1, we use 12.1
include_directories(/usr/local/cuda/include)

add_subdirectory(utils/nucleus nd)

add_library(link STATIC
    graph/match_gpu.h graph/match_gpu.cu
    kernels/indexing.h kernels/indexing.cu
    kernels/enumeration.h kernels/enumeration.cu
    kernels/gamma_enumeration.h kernels/gamma_enumeration.cu
    kernels/ours_enumeration.h kernels/ours_enumeration.cu
    kernels/enumeration_dfs.cu kernels/enumeration_bfs.cu
    kernels/cartesian_product.h kernels/cartesian_product.cu
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/cuda_helpers.h
    utils/automorphism.h utils/automorphism.cpp)

set_target_properties(link PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(link PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    -Xptxas=\"-v\"
    -DTHRUST_IGNORE_CUB_VERSION_CHECK
    >)


# executable: csmg

add_executable(csmg
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main.cpp)

set_target_properties(csmg PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(csmg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    -Xptxas=\"-v\"
    -DTHRUST_IGNORE_CUB_VERSION_CHECK
    >  $<$<COMPILE_LANGUAGE:CXX>:-Wall>)

target_link_libraries(csmg PUBLIC link)
target_link_libraries(csmg PUBLIC nd)



# executable: csmg_nbrSize1-5gb_resSpace2gb_sizeSpace1gb

add_executable(csmg_nbrSize1-5gb_resSpace2gb_sizeSpace1gb
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main.cpp)

set_target_properties(csmg_nbrSize1-5gb_resSpace2gb_sizeSpace1gb PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(csmg_nbrSize1-5gb_resSpace2gb_sizeSpace1gb PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    -Xptxas=\"-v\"
    -DTHRUST_IGNORE_CUB_VERSION_CHECK
    >  $<$<COMPILE_LANGUAGE:CXX>:-Wall>)

target_compile_definitions(csmg_nbrSize1-5gb_resSpace2gb_sizeSpace1gb PRIVATE
    NBR_SPACE=402653184  # 1.5 GB of uint32_t
    RES_SPACE=536870912  # 2 GB of uint32_t
    SIZE_SPACE=134217728)  # 1 GB of long 

target_link_libraries(csmg_nbrSize1-5gb_resSpace2gb_sizeSpace1gb PUBLIC link)
target_link_libraries(csmg_nbrSize1-5gb_resSpace2gb_sizeSpace1gb PUBLIC nd)



# executable: csmg_nbrSize1-5gb_resSpace3gb

add_executable(csmg_nbrSize1-5gb_resSpace3gb
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main.cpp)

set_target_properties(csmg_nbrSize1-5gb_resSpace3gb PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(csmg_nbrSize1-5gb_resSpace3gb PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    -Xptxas=\"-v\"
    -DTHRUST_IGNORE_CUB_VERSION_CHECK
    >  $<$<COMPILE_LANGUAGE:CXX>:-Wall>)

target_compile_definitions(csmg_nbrSize1-5gb_resSpace3gb PRIVATE
    NBR_SPACE=402653184  # 1.5 GB of uint32_t
    RES_SPACE=805306368  # 3 GB of uint32_t
    SIZE_SPACE=268435456)  # 2 GB of long 

target_link_libraries(csmg_nbrSize1-5gb_resSpace3gb PUBLIC link)
target_link_libraries(csmg_nbrSize1-5gb_resSpace3gb PUBLIC nd)


# end of file