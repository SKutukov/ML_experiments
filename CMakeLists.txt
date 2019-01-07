cmake_minimum_required(VERSION 3.0.0)

project(FactorMachine VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

include_directories(include)
add_executable(${PROJECT_NAME}
    "main.cpp"
    "src/factormachine.cpp"
    "include/factormachine.h")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
