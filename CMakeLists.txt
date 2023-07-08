cmake_minimum_required(VERSION 3.11-3.18)

project(grabcut)

find_package( OpenCV REQUIRED )
add_subdirectory(segmentation)
add_subdirectory(maxflow)
add_executable(grabcut main.cpp)

target_link_libraries(grabcut ${OpenCV_LIBS} segmentation)
target_include_directories(grabcut PUBLIC
                          "${OpenCV_INCLUDE_DIRS}"
                          "${PROJECT_SOURCE_DIR}/segmentation"
        )