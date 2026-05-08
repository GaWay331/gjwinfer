add_executable(test_tensor test_tensor.cpp)

target_link_libraries(test_tensor 
                    PRIVATE
                    gjwinfer
                    gtest_main
)

add_test(NAME TestTensor COMMAND test_tensor)