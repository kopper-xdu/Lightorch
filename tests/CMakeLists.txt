set(TEST_TGT ${CMAKE_PROJECT_NAME}_test)
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

find_package(GTest CONFIG REQUIRED)

add_executable(${TEST_TGT} ${TEST_SOURCES})

target_include_directories(${TEST_TGT} PUBLIC ../include)

target_link_libraries(${TEST_TGT} PRIVATE ${CMAKE_PROJECT_NAME} GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

add_test(NAME AllTestsInMain COMMAND ${TEST_TGT})