find_package(Boost COMPONENTS unit_test_framework REQUIRED)

include_directories(SYSTEM 
                    ${PROJECT_SOURCE_DIR}/src 
                    ${Boost_INCLUDE_DIRS} 
					${EIGEN_INCLUDE_DIR})

add_definitions(-D BOOST_TEST_DYN_LINK)

set(TESTS
	KMeansTest	
    )

foreach(test ${TESTS})
	add_executable(${test} ${test}.cpp)
	target_link_libraries(${test} kmeans ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
	add_test(NAME ${test} COMMAND ${test})
endforeach()

