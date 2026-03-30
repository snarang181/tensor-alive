cmake_minimum_required(VERSION 3.20)
project(tensor-alive LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Z3 - try cmake config first, fall back to manual
find_package(Z3 CONFIG QUIET)
if (NOT Z3_FOUND)
    # Manual fallback for Homebrew Z3
    find_library(Z3_LIB z3 HINTS /opt/homebrew/lib)
    find_path(Z3_INCLUDE z3++.h HINTS /opt/homebrew/include)
    if (Z3_LIB AND Z3_INCLUDE)
        message(STATUS "Found Z3 manually: ${Z3_LIB}")
        add_library(z3::z3 UNKNOWN IMPORTED)
        set_target_properties(z3::z3 PROPERTIES
            IMPORTED_LOCATION ${Z3_LIB}
            INTERFACE_INCLUDE_DIRECTORIES ${Z3_INCLUDE}
        )
        set(Z3_TARGET z3::z3)
    else()
        message(FATAL_ERROR "Z3 not found. Install with: brew install z3")
    endif()
else()
    message(STATUS "Found Z3 via cmake config")
    set(Z3_TARGET z3::libz3)
endif()

add_executable(tensor-alive
    src/main.cpp
    src/parser/Lexer.cpp
    src/parser/Parser.cpp
    src/smt/Z3Context.cpp
    src/smt/TensorEncoder.cpp
    src/smt/OpEncoder.cpp
    src/smt/ProgramEncoder.cpp
    src/checker/EquivalenceChecker.cpp
)

target_include_directories(tensor-alive PRIVATE src)
target_link_libraries(tensor-alive PRIVATE ${Z3_TARGET})
