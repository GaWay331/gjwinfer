cmake_minimum_required(VERSION 3.12)
project(gjwInfer CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Protobuf REQUIRED)
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG        v0.6.0
) 
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "" FORCE) # 轻量化使用，不需要 gflags
FetchContent_MakeAvailable(glog)
add_subdirectory(src)

enable_testing()
add_subdirectory(third_party/googletest)
add_subdirectory(tests)