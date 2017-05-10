# all modules that aren't executables go here
add_library(kmeans SHARED
            KMeans
           )

# external include directories
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

# libraries to link into the corso_rexarm.so we're creating
target_link_libraries(kmeans m stdc++)

# Following is for code that is for an actual executable
#set(EXES
#   )
#
#foreach(exe ${EXES})
#	add_executable(${exe} ${exe}.cpp)
#	target_link_libraries(${exe} kmeans m stdc++)
#endforeach()
