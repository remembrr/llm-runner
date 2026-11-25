cmake_minimum_required(VERSION 3.16)
project(llm-runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Where llama.cpp is installed. Default is the Homebrew prefix on this machine;
# override on other machines with:  cmake -B build -DLLAMA_PREFIX=$(brew --prefix llama.cpp)
# or point it at a source build's install tree.
set(LLAMA_PREFIX "/opt/homebrew/opt/llama.cpp" CACHE PATH "llama.cpp install prefix")
# ggml is a separate package; llama.h includes ggml.h. Default to the Homebrew prefix.
set(GGML_PREFIX  "/opt/homebrew/opt/ggml"      CACHE PATH "ggml install prefix")

if(NOT EXISTS "${LLAMA_PREFIX}/include/llama.h")
    message(FATAL_ERROR "llama.h not found under LLAMA_PREFIX='${LLAMA_PREFIX}'. "
                        "Set -DLLAMA_PREFIX=$(brew --prefix llama.cpp) or your install path.")
endif()

# Locate ggml.h: alongside llama.h, under GGML_PREFIX, or the default include dir.
find_path(GGML_INCLUDE_DIR ggml.h
          PATHS "${LLAMA_PREFIX}/include" "${GGML_PREFIX}/include" /opt/homebrew/include
          REQUIRED)

find_library(LLAMA_LIB NAMES llama PATHS "${LLAMA_PREFIX}/lib" REQUIRED)

add_executable(llm-runner main.cpp)
target_include_directories(llm-runner PRIVATE "${LLAMA_PREFIX}/include" "${GGML_INCLUDE_DIR}")
target_link_libraries(llm-runner PRIVATE "${LLAMA_LIB}")

# Find libllama (and its ggml deps in the same dir) at runtime without DYLD_* env vars.
set_target_properties(llm-runner PROPERTIES
    BUILD_RPATH   "${LLAMA_PREFIX}/lib"
    INSTALL_RPATH "${LLAMA_PREFIX}/lib")
