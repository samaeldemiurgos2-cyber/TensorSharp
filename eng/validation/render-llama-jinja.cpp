// Standalone renderer for the unchanged llama.cpp Jinja implementation.
// Compile with the checkout's common/jinja/{lexer,parser,runtime,value,string,caps}.cpp
// plus common/{json,unicode}.cpp, include paths common/, src/, vendor/, ggml/include/,
// and link ggml-base (for JSON assertion handling). No model or source edits needed.
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"
#include "json.h"
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: render-llama-jinja TEMPLATE INPUT_JSON");
        std::ifstream source_file(argv[1], std::ios::binary), input_file(argv[2]);
        if (!source_file || !input_file) throw std::runtime_error("Cannot open input file");
        std::string source((std::istreambuf_iterator<char>(source_file)), {});
        std::string input_text((std::istreambuf_iterator<char>(input_file)), {});
        common_json input = common_json::parse(input_text);
        jinja::lexer lexer;
        auto ast = jinja::parse_from_tokens(lexer.tokenize(source));
        jinja::caps_get(ast);
        jinja::context context(source);
        jinja::global_from_json(context, input, true);
        jinja::runtime runtime(context);
        std::cout << jinja::runtime::gather_string_parts(runtime.execute(ast))->as_string().str();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
