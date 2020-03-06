cmake_minimum_required(VERSION 3.1.0)
project(Fart VERSION 0.1.0)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

add_executable(Fart src/fart.cpp)
target_link_libraries(Fart ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
