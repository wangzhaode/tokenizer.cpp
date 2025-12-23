# tokenizer.cpp

![License](https://img.shields.io/badge/license-Apache%20License%202.0-green)
![Build Status](https://github.com/wangzhaode/tokenizer.cpp/actions/workflows/build.yml/badge.svg)
[![中文版本](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-green)](README_CN.md)

一个生产级、高兼容性的 C++ 分词器库，旨在直接支持 HuggingFace 模型的部署。

本项目提供了一个用于现代 LLM 分词流水线的高性能 C++ 实现，支持 BPE、WordPiece 和 Unigram 模型，并完美兼容复杂的 Normalization 和 Pre-tokenization 规则（如 GPT-2/Llama 3 的正则切分）。

## 主要特性

- **HuggingFace 兼容**：直接加载标准的 `tokenizer.json` 文件。
- **双 JSON 后端**：通过 `ujson` 桥接同时支持 `nlohmann/json` 和 `RapidJSON`。
- **高效高性能**：优化的 C++ 实现，使用 RapidJSON 后端可提升约 2 倍加载速度。
- **轻量自包含**：内置裁剪版 Oniguruma，最小化二进制体积。

## 支持模型

已在以下模型上验证并实现 100% 输出一致：
- **Llama 2 / 3 / 3.1 / 3.2**
- **Qwen 2.5 / 3 / 2.5-Coder**
- **DeepSeek V2 / V3 / R1**
- **Phi-3.5**
- **GLM-4**
- **Mistral**
- **Gemma**
- 更多...

## 集成

### 编译说明

**前置要求**:
- CMake 3.10+
- 支持 C++11 的编译器

```bash
mkdir build
cd build
# 默认：使用 nlohmann/json
cmake ..
# 可选：使用 RapidJSON 提升 2 倍加载速度
cmake .. -DUJSON_USE_RAPIDJSON=ON
make
```

### 运行测试

项目包含一套完整的测试套件，通过对比 HuggingFace `tokenizers` 的输出来验证正确性。

```bash
./test_main
```

## 使用示例

### 基础分词

```cpp
#include "tokenizer.hpp"
#include <iostream>

int main() {
    // 从 HuggingFace 风格的 tokenizer
    auto tokenizer = tokenizer::AutoTokenizer::from_pretrained("path/to/tokenizer/dir");

    std::string prompt = "Hello, world!";

    // 编码 (Encode)
    std::vector<int> ids = tokenizer->encode(prompt);

    // 解码 (Decode)
    std::string decoded = tokenizer->decode(ids);

    std::cout << "Encoded IDs: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << "\nDecoded: " << decoded << std::endl;

    return 0;
}
```

## 性能测试

本库针对加载速度进行了深度优化，特别是在处理超大模型配置文件时。使用 `RapidJSON` 后端可获得显著性能提升：

| 指标 (41 个模型 / 1691 个测试用例) | nlohmann/json | RapidJSON (via ujson) | 加速比 |
| :--- | :--- | :--- | :--- |
| **总计加载时间** | 92.40 s | 47.13 s | **1.96x** |
| **总计编码时间** | 0.25 s | 0.23 s | 1.07x |
| **总计总耗时** | 92.65 s | 47.36 s | **1.95x** |

*测试涵盖 41 种不同模型架构。*

## 文档

关于项目架构和技术实现的深度解析，请参阅技术文档 [doc/implementation_details_CN.md](doc/implementation_details_CN.md)。

## 许可证

Apache License 2.0。详情请见 [LICENSE](LICENSE) 文件。
