cmake_minimum_required(VERSION 3.16)
project(dsp_toolbox_c LANGUAGES C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_executable(demo_gain
    examples/demo_gain.c
    src/audio_block.c
)

target_include_directories(demo_gain PRIVATE include)

add_library(dsp_fir SHARED src/fir.c)
target_include_directories(dsp_fir PUBLIC include)
set_target_properties(dsp_fir PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
    NAME fir_golden
    COMMAND "${CMAKE_COMMAND}" -E env
        "DSP_FIR_LIB=$<TARGET_FILE:dsp_fir>"
        "${Python3_EXECUTABLE}" -m pytest
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_c_fir.py" -q
)
