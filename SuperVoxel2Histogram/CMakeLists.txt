cmake_minimum_required(VERSION 3.12)
project(SuperVoxel2Histogram)

add_executable(hist)
target_sources(hist PRIVATE "main.cpp")
install(TARGETS hist RUNTIME DESTINATION .)