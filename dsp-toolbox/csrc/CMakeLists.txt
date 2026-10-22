cmake_minimum_required(VERSION 3.16)
project(gain_demo LANGUAGES C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_executable(demo_gain
    src/demo_gain.c
    src/audio_block.c
)

target_include_directories(demo_gain PRIVATE include)
add_executable(demo_remove_element
    src/demo_remove_element.c)
add_executable(demo_reverse_string
    src/demo_reverse_string.c)
add_executable(demo_sorted_squares 
    src/demo_sorted_squares.c)