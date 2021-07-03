cmake_minimum_required(VERSION 3.12)
project(SLIC3DSuperVoxel)
set(SRC "graph_segmentation.cpp"
		"main.cpp"
		"SLIC3D.cpp"
		"SourceVolume.cpp")
add_executable(slic3d)
target_sources(slic3d PRIVATE "${SRC}")
install(TARGETS slic3d RUNTIME DESTINATION .)