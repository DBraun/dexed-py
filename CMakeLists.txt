cmake_minimum_required(VERSION 3.15...3.29)
project(dexed)

# Allow Python executable to be specified via command line
if(DEFINED Python_EXECUTABLE)
    set(Python_ROOT_DIR "")
    set(Python_FIND_STRATEGY LOCATION)
endif()

# Find Python and nanobind
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

# Check if nanobind submodule is present
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nanobind/CMakeLists.txt")
    message(FATAL_ERROR "Nanobind submodule not found. Please run: git submodule update --init --recursive")
endif()

# Disable nanobind tests and extras to speed up configuration
set(NB_TEST OFF CACHE INTERNAL "")
set(NB_TEST_STABLE_ABI OFF CACHE INTERNAL "")
set(NB_TEST_SHARED_BUILD OFF CACHE INTERNAL "")

# Add nanobind from submodule
add_subdirectory(nanobind)

# Dexed source files (excluding tuning.cc which has JUCE dependencies)
set(DEXED_SOURCES
    src/msfa/dx7note.cc
    src/msfa/env.cc
    src/msfa/exp2.cc
    src/msfa/fm_core.cc
    src/msfa/fm_op_kernel.cc
    src/msfa/freqlut.cc
    src/msfa/lfo.cc
    src/msfa/pitchenv.cc
    src/msfa/sin.cc
    src/EngineMkI.cpp
    src/EngineOpl.cpp
)

# Create the Python module
# Use a different name for the compiled module to avoid naming conflicts
nanobind_add_module(
    _dexed
    src/dexed.cpp
    ${DEXED_SOURCES}
)

# Set C++ standard
target_compile_features(_dexed PRIVATE cxx_std_17)

# Include directories
target_include_directories(_dexed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/msfa
)

# nanobind_add_module already handles the linking

# Installation rules for scikit-build-core
install(TARGETS _dexed LIBRARY DESTINATION dexed COMPONENT python_modules)

# Install Python package files
install(FILES
    dexed/__init__.py
    dexed/version.py
    dexed/algorithms.py
    dexed/graph.py
    dexed/patch.py
    dexed/preset.py
    dexed/synth.py
    DESTINATION dexed
    COMPONENT python_modules
)