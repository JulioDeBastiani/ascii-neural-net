cmake_minimum_required(VERSION 3.13)
project(proj-ascii-neural-net)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(ascii-neural-net ascii-neural-net/main ascii-neural-net/t)
target_include_directories(ascii-neural-net PRIVATE .)

# add_subdirectory(ascii-neural-net)

target_link_libraries(ascii-neural-net Eigen3::Eigen)