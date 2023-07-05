cmake_minimum_required(VERSION 3.11-3.18)

project(main)

find_package( OpenCV REQUIRED )
add_subdirectory(maxflow)
add_executable(main main.cpp)

target_link_libraries(main ${OpenCV_LIBS} maxflow)
target_include_directories(main PUBLIC
                          "${OpenCV_INCLUDE_DIRS}"
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/maxflow"
                          )