cmake_minimum_required(VERSION 3.25)
project(recommender_system)

set(CMAKE_CXX_STANDARD 20)

find_package(indicators CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

add_executable(
        recommender_system
        main.cpp
        core.cpp
)

target_link_libraries(
        recommender_system
        PRIVATE
        indicators::indicators
        cxxopts::cxxopts
)
