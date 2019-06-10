cmake_minimum_required(VERSION 3.0.0)

project(FactorMachine VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC --pedantic -O3 -Wall -Wextra -std=c++17")
include_directories(include)
add_executable(${PROJECT_NAME}
    "main.cpp"
    "src/factormachine.cpp"
    "include/factormachine.h"
    "boost_serialization_eigen.h")
find_package (Eigen3 3.3  REQUIRED NO_MODULE)
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options regex serialization REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${Boost_LIBRARIES})
