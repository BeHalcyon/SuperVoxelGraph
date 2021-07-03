cmake_minimum_required(VERSION 3.12)
project(LabelSuperVoxelQt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)


add_executable(labelqt)
target_link_libraries(labelqt Qt5::Widgets)
target_sources(labelqt PRIVATE "main.cpp"
								"ParameterControlWidget.cpp"
								"SliceView.cpp"
								"VolumeToLabel.cpp"
								"ParameterControlWidget.ui"
								"VolumeToLabel.ui"
								"VolumeToLabel.qrc")

install(TARGETS labelqt DESTINATION bin)