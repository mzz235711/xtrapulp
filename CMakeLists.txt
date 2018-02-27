project(grape C CXX)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(xtrapulp DIR_SRCS)
add_executable(xtrapulp ${DIR_SRCS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3 -Wall")

find_package(MPI)
include_directories(${MPI_CXX_INCLUDE_PATH})

target_link_libraries(xtrapulp ${MPI_CXX_LIBRARIES})

