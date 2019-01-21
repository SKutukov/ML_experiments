cmake_minimum_required(VERSION 3.0.0)

project(AffinityPropogation VERSION 0.1 LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")


find_package (Eigen3  REQUIRED)

include_directories(include)
add_executable(${PROJECT_NAME}
    "main.cpp"
    "src/afPropogation.cpp"
    "include/afPropogation.h")


include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME})
