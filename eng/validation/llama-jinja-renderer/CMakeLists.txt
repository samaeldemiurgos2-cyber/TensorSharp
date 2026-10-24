cmake_minimum_required(VERSION 3.20)
project(render_llama_jinja LANGUAGES CXX)

# Example: cmake -S eng/validation/llama-jinja-renderer -B artifacts/jinja \
#   -DLLAMA_SOURCE_DIR=/path/to/llama.cpp \
#   -DGGML_BASE_LIBRARY=/path/to/llama.cpp/build/ggml/src/libggml-base.so
# cmake --build artifacts/jinja --config Release
# This reads an existing checkout and links its built assertion handler; it never
# fetches, patches, configures or builds upstream sources in their original tree.
set(LLAMA_SOURCE_DIR "" CACHE PATH "Unchanged llama.cpp source checkout")
set(GGML_BASE_LIBRARY "" CACHE FILEPATH "Existing ggml-base library/import library")
if(NOT EXISTS "${LLAMA_SOURCE_DIR}/common/jinja/runtime.cpp" OR NOT EXISTS "${GGML_BASE_LIBRARY}")
    message(FATAL_ERROR "Provide LLAMA_SOURCE_DIR and an existing GGML_BASE_LIBRARY")
endif()
add_executable(render-llama-jinja ../render-llama-jinja.cpp
    ${LLAMA_SOURCE_DIR}/common/jinja/lexer.cpp
    ${LLAMA_SOURCE_DIR}/common/jinja/parser.cpp
    ${LLAMA_SOURCE_DIR}/common/jinja/runtime.cpp
    ${LLAMA_SOURCE_DIR}/common/jinja/value.cpp
    ${LLAMA_SOURCE_DIR}/common/jinja/string.cpp
    ${LLAMA_SOURCE_DIR}/common/jinja/caps.cpp
    ${LLAMA_SOURCE_DIR}/common/json.cpp
    ${LLAMA_SOURCE_DIR}/common/unicode.cpp)
target_compile_features(render-llama-jinja PRIVATE cxx_std_17)
target_include_directories(render-llama-jinja PRIVATE
    ${LLAMA_SOURCE_DIR}/common ${LLAMA_SOURCE_DIR}/src
    ${LLAMA_SOURCE_DIR}/vendor ${LLAMA_SOURCE_DIR}/ggml/include)
target_link_libraries(render-llama-jinja PRIVATE "${GGML_BASE_LIBRARY}")
