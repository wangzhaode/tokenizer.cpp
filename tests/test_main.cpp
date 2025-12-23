/**
 * test_main.cpp - Tokenizer Test
 *
 * 遍历 tests/models/ 目录下的所有模型，加载 tokenizer 并运行 test_cases.jsonl 测试
 *
 * 用法: ./test_main [model_filter]
 *   model_filter: 可选，用于筛选特定模型 (如 "Qwen")
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <dirent.h>
#include <sys/stat.h>
#include "tokenizer.hpp"

#include <utf8proc/utf8proc.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ==================== 颜色定义 ====================
namespace Color {
    const std::string RESET   = "\033[0m";
    const std::string RED     = "\033[31m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string CYAN    = "\033[36m";
    const std::string BOLD    = "\033[1m";
    const std::string GREY    = "\033[90m";
}

// 计算字符串在终端显示的视觉宽度，跳过 ANSI 转义序列，处理 ZWJ Emoji 序列
int get_display_width(const std::string& str) {
    int width = 0;
    const uint8_t* ptr = (const uint8_t*)str.c_str();
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t len;
    bool last_was_zwj = false;

    while (true) {
        if (*ptr == '\0') break;

        // 跳过 ANSI 转义序列 (如 \033[90m)
        if (*ptr == '\033' && *(ptr+1) == '[') {
            ptr += 2;
            while (*ptr != '\0' && !isalpha(*ptr)) ptr++;
            if (*ptr != '\0') ptr++;
            continue;
        }

        len = utf8proc_iterate(ptr, -1, &codepoint);
        if (len <= 0 || codepoint == -1) break;

        int w = utf8proc_charwidth(codepoint);
        if (w < 0) w = 0;

        if (last_was_zwj) {
            last_was_zwj = false;
        } else if (codepoint == 0x200D) {
            last_was_zwj = true;
        } else {
            width += w;
        }
        ptr += len;
    }
    return width;
}

void print_aligned(const std::string& str, int target_width) {
    int current_width = get_display_width(str);
    std::cout << str;
    if (target_width > current_width) {
        for (int i = 0; i < target_width - current_width; ++i) {
            std::cout << " ";
        }
    }
}

// ==================== 工具函数 ====================

std::string visualize(const std::string& input) {
    std::string out;
    for (char c : input) {
        if (c == '\n') {
            out += Color::GREY + "\\n" + Color::RESET + "\n";
        } else if (c == '\r') {
            out += Color::GREY + "\\r" + Color::RESET;
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
    return out;
}

std::vector<std::string> list_model_dirs(const std::string& models_path) {
    std::vector<std::string> dirs;
    DIR* dir = opendir(models_path.c_str());
    if (!dir) {
        std::cerr << Color::RED << "❌ Cannot open models directory: " << models_path << Color::RESET << std::endl;
        return dirs;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        std::string full_path = models_path + "/" + name;
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            dirs.push_back(name);
        }
    }
    closedir(dir);

    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

// ==================== 测试用例运行器 ====================

struct TestResult {
    int passed = 0;
    int failed = 0;
    int skipped = 0;
};

// 运行 basic 类型测试 (纯 tokenization + decode)
bool run_basic_test(tokenizer::PreTrainedTokenizer* tok, const json& test_case, bool verbose = false) {
    std::string input = test_case["input"];
    std::vector<int> expected_ids = test_case["ids_raw"].get<std::vector<int>>();

    // 1. 测试 Encode
    std::vector<int> result = tok->encode(input, false);
    bool ids_match = (result == expected_ids);

    // 2. 测试 Decode
    std::string decoded_text = tok->decode(expected_ids);
    std::string expected_decode = input;
    if (test_case.contains("decoded_full")) {
        expected_decode = test_case["decoded_full"];
    }
    bool decode_match = (decoded_text == expected_decode);

    if (ids_match && decode_match) {
        return true;
    } else {
        if (verbose) {
            std::cout << std::endl << Color::GREY << "     ┌── Input ──────────────────────────────────────" << Color::RESET << std::endl;
            std::cout << "     │ " << "#" << visualize(input) << "#" << std::endl;

            if (!ids_match) {
                std::cout << Color::RED << "     ├── IDs Mismatch ❌" << Color::RESET << std::endl;
                std::cout << Color::GREY << "     │ Expected: ";
                for (int id : expected_ids) std::cout << id << " ";
                std::cout << std::endl << "     │ Got:      ";
                for (int id : result) std::cout << id << " ";
                std::cout << Color::RESET << std::endl;
            }

            if (!decode_match) {
                std::cout << Color::RED << "     ├── Decode Mismatch ❌" << Color::RESET << std::endl;
                std::cout << Color::GREY << "     │ Expected: " << Color::RESET << "#" << visualize(expected_decode) << "#" << std::endl;
                std::cout << Color::GREY << "     │ Decoded:  " << Color::RESET << "#" << visualize(decoded_text) << "#" << std::endl;
            }

            std::cout << Color::GREY << "     └──────────────────────────────────────────────────" << Color::RESET << std::endl;
        }
        return false;
    }
}

// 运行 chat 类型测试 (apply_chat_template)
bool run_chat_test(tokenizer::PreTrainedTokenizer* tok, const json& test_case, bool verbose = false) {
    std::string name = test_case["name"];
    std::string expected_text = test_case["formatted_text"];
    std::vector<int> expected_ids = test_case["ids"].get<std::vector<int>>();
    bool add_gen_prompt = test_case.value("add_generation_prompt", false);

    std::string result_text; // Declare result_text
    tokenizer::ChatMessages messages; // Declare messages
    bool has_complex = false;
    if (test_case["messages"].is_array()) {
        for (const auto& msg : test_case["messages"]) {
            if (msg.is_object() && msg.contains("role")) {
                messages.push_back({msg["role"], msg.value("content", "")});
                if (msg.size() > 2 || (msg.size() == 2 && !msg.contains("content"))) has_complex = true;
            }
        }
    }
    if (has_complex) {
        result_text = tok->apply_chat_template(test_case["messages"].dump(), add_gen_prompt);
    } else {
        result_text = tok->apply_chat_template(messages, add_gen_prompt);
    }

    // 1. 比较生成的文本
    bool text_match = (result_text == expected_text);

    // 2. 比较生成的 Tokens
    std::vector<int> result_ids = tok->encode(result_text, false);
    bool ids_match = (result_ids == expected_ids);

    if (text_match && ids_match) {
        return true;
    } else {
        if (verbose) {
            if (!text_match) {
                std::cout << Color::RED << "     ├── Text Mismatch ❌" << Color::RESET << std::endl;
                std::cout << Color::GREY << "     │ Expected: " << Color::RESET << visualize(expected_text) << std::endl;
                std::cout << Color::GREY << "     │ Actual:   " << Color::RESET << visualize(result_text) << std::endl;
            } else {
                std::cout << std::endl;
                std::cout << Color::GREY << "     │ Expected: " << Color::RESET << visualize(expected_text) << std::endl;
                std::cout << Color::GREY << "     │ Actual:   " << Color::RESET << visualize(result_text) << std::endl;
            }

            if (!ids_match) {
                std::cout << Color::RED << "     ├── Token IDs Mismatch ❌" << Color::RESET << std::endl;
                std::cout << Color::GREY << "     │ Expected: ";
                for (int id : expected_ids) std::cout << id << " ";
                std::cout << std::endl << "     │ Got:      ";
                for (int id : result_ids) std::cout << id << " ";
                std::cout << Color::RESET << std::endl;
            }
            std::cout << Color::GREY << "     └──────────────────────────────────────────────────" << Color::RESET << std::endl;
        }
        return false;
    }
}

// 运行单个模型的所有测试
TestResult run_model_tests(const std::string& model_path, const std::string& model_name, bool verbose = false) {
    TestResult result;

    // 1. 加载 tokenizer
    auto tok = tokenizer::AutoTokenizer::from_pretrained(model_path);
    if (!tok) {
        std::cout << Color::RED << "  ❌ Failed to load tokenizer" << Color::RESET << std::endl;
        return result;
    }

    // 2. 加载 test_cases.jsonl
    std::string cases_path = model_path + "/test_cases.jsonl";
    std::ifstream f(cases_path);
    if (!f.is_open()) {
        std::cout << Color::YELLOW << "  ⚠️  No test_cases.jsonl found" << Color::RESET << std::endl;
        return result;
    }

    // 3. 逐行读取并测试
    std::string line;
    int case_num = 0;

    while (std::getline(f, line)) {
        if (line.empty()) continue;

        json test_case;
        try {
            test_case = json::parse(line);
        } catch (const std::exception& e) {
            std::cout << "  ⚠️  JSON parse error at line " << case_num + 1 << std::endl;
            result.skipped++;
            continue;
        }

        case_num++;
        std::string type = test_case.value("type", "basic");
        std::string desc;

        if (type == "basic") {
            std::string input = test_case.value("input", "");
            std::string clean_input;
            for (char c : input) {
                if (c == '\n') clean_input += Color::GREY + "\\n" + Color::RESET;
                else if (c == '\r') clean_input += Color::GREY + "\\r" + Color::RESET;
                else if (c == '\t') clean_input += Color::GREY + "\\t" + Color::RESET;
                else clean_input += c;
            }

            const int max_w = 32;
            const int truncate_w = max_w - 3;
            int current_w = 0;
            const uint8_t* ptr = (const uint8_t*)clean_input.c_str();
            utf8proc_int32_t codepoint;
            utf8proc_ssize_t len;
            size_t bytes_len = 0;
            bool last_was_zwj = false;

            while ((len = utf8proc_iterate(ptr, -1, &codepoint)) > 0) {
                if (codepoint == -1) break;
                int w = utf8proc_charwidth(codepoint);
                if (w < 0) w = 0;

                int added_w = 0;
                if (last_was_zwj) {
                    last_was_zwj = false;
                } else if (codepoint == 0x200D) {
                    last_was_zwj = true;
                } else {
                    added_w = w;
                }

                if (current_w + added_w > max_w) break;
                current_w += added_w;
                ptr += len;
                bytes_len += len;
            }

            if (bytes_len < clean_input.length()) {
                current_w = 0;
                ptr = (const uint8_t*)clean_input.c_str();
                bytes_len = 0;
                last_was_zwj = false;
                while ((len = utf8proc_iterate(ptr, -1, &codepoint)) > 0) {
                    int w = utf8proc_charwidth(codepoint);
                    if (w < 0) w = 0;

                    int added_w = 0;
                    if (last_was_zwj) {
                        last_was_zwj = false;
                    } else if (codepoint == 0x200D) {
                        last_was_zwj = true;
                    } else {
                        added_w = w;
                    }

                    if (current_w + added_w > truncate_w) break;
                    current_w += added_w;
                    ptr += len;
                    bytes_len += len;
                }
                desc = clean_input.substr(0, bytes_len) + "...";
            } else {
                desc = clean_input;
            }
        } else if (type == "chat") {
            desc = test_case.value("name", "unnamed");
        } else {
            result.skipped++;
            continue;
        }

        std::cout << "  ├─ " << std::left << std::setw(8) << ("[" + type + "]");
        print_aligned(desc, 45);

        bool passed = false;

        try {
            if (type == "basic") {
                passed = run_basic_test(tok.get(), test_case, verbose);
            } else if (type == "chat") {
                passed = run_chat_test(tok.get(), test_case, verbose);
            }
        } catch (const std::exception& e) {
            std::cout << Color::RED << "[ERROR]" << Color::RESET << std::endl;
            if (verbose) {
                std::cout << "     └─ " << e.what() << std::endl;
            }
            result.failed++;
            continue;
        }

        if (passed) {
            std::cout << Color::GREEN << "[PASS]" << Color::RESET << std::endl;
            result.passed++;
        } else {
            std::cout << Color::RED << "[FAIL]" << Color::RESET << std::endl;
            result.failed++;
        }
    }

    return result;
}

// ==================== 主函数 ====================

int main(int argc, char** argv) {
    std::string models_path = "../tests/models";
    std::string model_filter = "";
    bool verbose = true;  // 默认输出详细信息

    if (argc > 1) {
        models_path = argv[1];
    }
    if (argc > 2) {
        model_filter = argv[2];
    }

    std::cout << "📂 Models Directory: " << models_path << std::endl;
    if (!model_filter.empty()) {
        std::cout << "🔍 Filter: " << model_filter << std::endl;
    }

    // 获取所有模型目录
    std::vector<std::string> model_dirs = list_model_dirs(models_path);
    if (model_dirs.empty()) {
        std::cerr << "No models found!" << std::endl;
        return 1;
    }

    std::cout << "📋 Found " << model_dirs.size() << " model(s)\n" << std::endl;

    // 统计
    int total_models = 0;
    int total_passed = 0;
    int total_failed = 0;
    int total_skipped = 0;
    std::vector<std::string> failed_models;

    // 遍历每个模型
    for (const std::string& model_name : model_dirs) {
        // 应用过滤器
        if (!model_filter.empty() && model_name.find(model_filter) == std::string::npos) {
            continue;
        }

        total_models++;
        std::string model_path = models_path + "/" + model_name;

        std::cout << Color::BLUE << Color::BOLD << "┏━━ Model: " << model_name << Color::RESET << std::endl;

        TestResult result = run_model_tests(model_path, model_name, verbose);

        total_passed += result.passed;
        total_failed += result.failed;
        total_skipped += result.skipped;

        // 打印模型小结
        std::cout << "┗━━ ";
        if (result.failed == 0) {
            std::cout << Color::GREEN << "✓ " << result.passed << " passed";
        } else {
            std::cout << Color::RED << "✗ " << result.failed << " failed";
            failed_models.push_back(model_name);
        }
        if (result.skipped > 0) {
            std::cout << Color::YELLOW << ", " << result.skipped << " skipped";
        }
        std::cout << Color::RESET << std::endl << std::endl;
    }

    // 打印总结
    std::cout << "==================================================" << std::endl;
    std::cout << "               TEST SUMMARY                       " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << " Models Tested : " << total_models << std::endl;
    std::cout << " Total Cases   : " << (total_passed + total_failed + total_skipped) << std::endl;
    std::cout << Color::GREEN << " Passed        : " << total_passed << Color::RESET << std::endl;

    if (total_failed > 0) {
        std::cout << Color::RED << " Failed        : " << total_failed << Color::RESET << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << " Failed Models:" << std::endl;
        for (const auto& m : failed_models) {
            std::cout << Color::RED << "  - " << m << Color::RESET << std::endl;
        }
        return 1;
    } else {
        std::cout << Color::GREEN << " Failed        : 0" << Color::RESET << std::endl;
        std::cout << "\n✨ All tests passed! ✨" << std::endl;
        return 0;
    }
}
