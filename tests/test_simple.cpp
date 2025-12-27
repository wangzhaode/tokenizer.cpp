#include "tokenizer.hpp"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer_path>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    auto tokenizer = tokenizer::AutoTokenizer::from_pretrained(path);

    if (!tokenizer) {
        std::cerr << "Failed to load tokenizer from: " << path << std::endl;
        return 1;
    }

    std::string prompt = "<|im_start|>system\n"
            "你是一个专业的AI助手，请用中文回答用户的问题。<|im_end|>\n"
            "<|im_start|>user\n"
            "你好！你能介绍一下你自己吗？<|im_end|>\n"
            "<|im_start|>assistant\n";

    // 编码 (Encode)
    std::vector<int> ids = tokenizer->encode(prompt);

    // 解码 (Decode)
    std::string decoded = tokenizer->decode(ids);

    std::cout << "Encoded IDs: ";
    for (int id: ids) std::cout << id << " ";
    std::cout << std::endl;

    std::cout << "Decoded: " << decoded << std::endl;

    // Chat Template Test
    std::cout << "\n--- Chat Template Test ---\n";
    tokenizer::ChatMessages messages = {
        {"user", "Hello"},
        {"assistant", "Hi there!"}
    };

    std::string chat_output = tokenizer->apply_chat_template(messages, false);
    std::cout << "Chat Output:\n" << chat_output << std::endl;

    return 0;
}
