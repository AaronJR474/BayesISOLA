cmake_minimum_required(VERSION 3.17...4.4)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "BayesISOLA")
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES Fortran)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O0" CACHE STRING "" FORCE)
endif()

set(AXITRA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fortran/axitra")

add_executable(gr_xyz "${AXITRA_SOURCE_DIR}/gr_xyz.for")
add_executable(elemse "${AXITRA_SOURCE_DIR}/elemse.for")

target_include_directories(gr_xyz PRIVATE "${AXITRA_SOURCE_DIR}")
target_include_directories(elemse PRIVATE "${AXITRA_SOURCE_DIR}")

install(TARGETS gr_xyz elemse RUNTIME DESTINATION BayesISOLA/_bin)
