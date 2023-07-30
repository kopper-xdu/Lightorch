cmake_minimum_required(VERSION 3.10)

project(MyProject VERSION 1.0
                  DESCRIPTION "Very nice project"
                  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(test test.cpp src/tensor.cpp src/autograd.cpp src/ops/ops.cpp)

target_include_directories(test PUBLIC include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)
endif()