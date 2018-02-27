aux_source_directory(. srcs)
include_directories(.)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
add_library(xtrapulp SHARED ${srcs})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3 -Wall")
find_package(MPI)
include_directories(${MPI_CXX_INCLUDE_PATH})

target_link_libraries(xtrapulp ${MPI_CXX_LIBRARIES})

