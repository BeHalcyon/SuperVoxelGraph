cmake_minimum_required(VERSION 3.12)
project(VolumeGenerator)

add_executable(volgen)
target_sources(volgen PRIVATE "main.cpp")

install(TARGETS volgen RUNTIME DESTINATION .)