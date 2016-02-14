cmake_minimum_required(VERSION 2.8)
project(KMeans)

# Add path to search for modules
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

## Enfore C++11 compatability
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
	message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11
	support.  Please use a different C++ compiler.")
endif()

# -Wno-format-zero-length: permit printf("");
# -Wno-unused-parameter: permit a function to ignore an argument
# Setup CFLAGS
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wvla -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-format-zero-length -std=c99")

# Setup CXXFLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wvla -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-zero-length")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_AUTOMOC ON)

# Find required libs
find_package(Eigen3 REQUIRED)

# Dependency on rex_arm and libcorso_rexarm.so
add_subdirectory(src)

#enable_testing()
#add_subdirectory(test)

	
message(STATUS "")
message(STATUS "Building a ${CMAKE_BUILD_TYPE} build")
message(STATUS "")

