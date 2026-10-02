cmake_minimum_required(VERSION 3.20)
project(ye3t_methods_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(YE3T_RUNTIME_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../ye3t" CACHE PATH
    "YE3T source tree containing ye3t/runtime/csrc/ye3t_runtime_core.cpp")
option(YE3T_NATIVE_CPU "Compile CPU kernels for the build host" OFF)
option(YE3T_ENABLE_IPO "Enable interprocedural optimization" OFF)

if(NOT EXISTS "${YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc/ye3t_runtime_core.cpp")
  message(FATAL_ERROR "YE3T_RUNTIME_SOURCE must point to a YE3T source tree")
endif()
find_package(yaml-cpp CONFIG REQUIRED)

if(YE3T_NATIVE_CPU)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  else()
    message(FATAL_ERROR "YE3T_NATIVE_CPU requires GNU or Clang")
  endif()
endif()
if(YE3T_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ye3t_ipo_supported OUTPUT ye3t_ipo_error)
  if(NOT ye3t_ipo_supported)
    message(FATAL_ERROR "Interprocedural optimization unavailable: ${ye3t_ipo_error}")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

add_library(ye3t_runtime_core STATIC
  "${YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc/ye3t_runtime_core.cpp")
target_include_directories(ye3t_runtime_core PUBLIC
  "${YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc")

add_library(ye3t_methods_cpu STATIC
  src/ye3t_canonical_json_hash.cpp
  src/ye3t_lifted_cauchy_model.cpp
  src/ye3t_lifted_cauchy_cpu.cpp
  src/ye3t_lifted_cauchy_source.cpp
  src/ye3t_sha256.cpp
  src/ye3t_tagged_cauchy_model.cpp
  src/ye3t_tagged_cauchy_cpu.cpp
  src/ye3t_yace_model.cpp
  src/ye3t_cpu_evaluator.cpp
  src/ye3t_cpu_source_tiling.cpp)
target_include_directories(ye3t_methods_cpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(ye3t_methods_cpu PUBLIC ye3t_runtime_core yaml-cpp::yaml-cpp)

add_library(ye3t_tagged_c_api SHARED src/ye3t_tagged_c_api.cpp)
target_link_libraries(ye3t_tagged_c_api PRIVATE ye3t_methods_cpu)
install(TARGETS ye3t_tagged_c_api LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
