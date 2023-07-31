find_package(OpenMP)
find_package(spdlog CONFIG REQUIRED)

add_library(${CMAKE_PROJECT_NAME} STATIC tensor.cpp autograd.cpp ops/ops.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ../include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE spdlog::spdlog)