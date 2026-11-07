cmake_minimum_required(VERSION 3.20)

project(ReconstructionEna VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file"
    )
endif()

# The reconstruction stack uses CGAL's floating-point kernels, Eigen and
# OpenMP. ANN is distributed with this repository as a Windows library.
set(CGAL_DISABLE_GMP TRUE)
find_package(CGAL CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(utils/pqp)

add_executable(Reconstruction_Ena
    src/main.cpp
    src/BaseModel.cpp
    src/BooleanComputation.cpp
    src/Point3D.cpp
    src/Reconstruction_Basedon_Paricles_LBFGS.cpp
)

target_include_directories(Reconstruction_Ena PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/ann/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/lbfgs
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/pqp
)

target_compile_definitions(Reconstruction_Ena PRIVATE
    CGAL_DO_NOT_USE_MPZF
    NOMINMAX
    _HAS_STD_BYTE=0
    WIN32_LEAN_AND_MEAN
)

target_sources(Reconstruction_Ena PRIVATE
    utils/lbfgs/lbfgs.cpp
)

target_link_libraries(Reconstruction_Ena PRIVATE
    pqp
    CGAL::CGAL
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

if(WIN32)
    set(ANN_LIB "${CMAKE_CURRENT_SOURCE_DIR}/utils/ann/lib/ANN.lib")
    set(ANN_DLL "${CMAKE_CURRENT_SOURCE_DIR}/utils/ann/lib/ANN.dll")

    if(NOT EXISTS "${ANN_LIB}")
        message(FATAL_ERROR "ANN.lib was not found at ${ANN_LIB}")
    endif()

    add_library(ann SHARED IMPORTED)
    set_target_properties(ann PROPERTIES
        IMPORTED_IMPLIB "${ANN_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${CMAKE_CURRENT_SOURCE_DIR}/utils/ann/include"
    )

    if(EXISTS "${ANN_DLL}")
        set_target_properties(ann PROPERTIES IMPORTED_LOCATION "${ANN_DLL}")
    else()
        message(FATAL_ERROR "ANN.dll was not found at ${ANN_DLL}")
    endif()

    target_link_libraries(Reconstruction_Ena PRIVATE ann opengl32 glu32)

    add_custom_command(TARGET Reconstruction_Ena POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ANN_DLL}"
            "$<TARGET_FILE_DIR:Reconstruction_Ena>/ANN.dll"
        COMMENT "Copying ANN.dll next to Reconstruction_Ena.exe"
    )
endif()

if(MSVC)
    target_compile_options(Reconstruction_Ena PRIVATE /W3 /fp:fast /bigobj)
    target_compile_options(Reconstruction_Ena PRIVATE
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(Reconstruction_Ena PRIVATE /STACK:10000000)
endif()

set_target_properties(Reconstruction_Ena PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
