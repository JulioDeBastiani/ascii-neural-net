cmake_minimum_required(VERSION 3.10)
project(proj-ascii-neural-net)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED)

add_executable(ann
    ascii-neural-net/activation-functions
    ascii-neural-net/layers
    ascii-neural-net/main
    ascii-neural-net/model
    ascii-neural-net/status
    ascii-neural-net/types
)

target_include_directories(ann PRIVATE .)

target_link_libraries(ann Eigen3::Eigen)
target_link_libraries(ann ${Boost_LIBRARIES})