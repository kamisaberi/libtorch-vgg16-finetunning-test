cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(cpp_vgg16_finetune)

set(CMAKE_C_COMPILER "gcc-12")
set(CMAKE_CXX_COMPILER "g++-12")
set(CMAKE_CUDA_HOST_COMPILER "g++-12")
set(CMAKE_CUDA_ARCHITECTURES "75;86")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "Setting default build type to: ${CMAKE_BUILD_TYPE}")
endif()

list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)
find_package(Torch REQUIRED)
find_package(xTorch REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SndFile REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(TEST_LIBRARIES ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        OpenSSL::Crypto
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${XTORCH_LIBRARIES}
        xTorch::xTorch


)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
target_include_directories(main PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})
target_link_libraries(main ${TEST_LIBRARIES})
configure_file(vgg16_bn_weights.pt vgg16_bn_weights.pt  COPYONLY)
