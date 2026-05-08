file(GLOB_RECURSE SRC_FILES
     "core/*.cpp"
     "parser/*.cpp")
add_library(gjwinfer SHARED ${SRC_FILES})

target_include_directories(gjwinfer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_include_directories(gjwinfer PRIVATE ${CMAKE_SOURCE_DIR}/third_party/eigen3/)

target_link_libraries(gjwinfer PRIVATE protobuf::libprotobuf glog::glog)
