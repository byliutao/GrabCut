aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} segmentation_src)
find_package( OpenCV REQUIRED )

add_library(segmentation OBJECT ${segmentation_src})


target_link_libraries(segmentation ${OpenCV_LIBS} maxflow)
target_include_directories(segmentation PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
        "${PROJECT_SOURCE_DIR}/maxflow"
        )

