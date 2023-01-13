cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(aggupscaleop)

set(Torch_DIR D:/examples/torchfiles/cplusdow/libtorch-win-shared-with-deps-1.13.1+cpu/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(aggupscaleop aggupscaleop.cpp)
target_link_libraries(aggupscaleop "${TORCH_LIBRARIES}")
set_property(TARGET aggupscaleop PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET aggupscaleop
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:aggupscaleop>)
endif (MSVC)