cmake_minimum_required(VERSION 3.12)
project(SuperVoxelGraph)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/workspace")

add_subdirectory(SLIC3DSuperVoxel) # slic3d
add_subdirectory(SuperVoxel2Histogram)
add_subdirectory(VolumeGenerator)
option(SV_BUILD_QT_LABEL_TOOL "OFF")
if(SV_BUILD_QT_LABEL_TOOL)
add_subdirectory(LabelSuperVoxelQt)
endif()