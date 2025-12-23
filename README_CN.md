# tokenizer.cpp

![License](https://img.shields.io/badge/license-Apache%20License%202.0-green)
![Build Status](https://github.com/wangzhaode/tokenizer.cpp/actions/workflows/build.yml/badge.svg)
[![中文版本](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-green)](README_CN.md)

一个生产级、高兼容性的 C++ 分词器库，旨在直接支持 HuggingFace 模型的部署。

本项目提供了一个用于现代 LLM 分词流水线的高性能 C++ 实现，支持 BPE、WordPiece 和 Unigram 模型，并完美兼容复杂的 Normalization 和 Pre-tokenization 规则（如 GPT-2/Llama 3 的正则切分）。

## 主要特性

- **HuggingFace 兼容**：直接加载标准的 `tokenizer.json` 文件，无需转换。
- **全面支持**：支持 BPE (Byte-Pair Encoding), WordPiece, Unigram 算法。
- **复杂的规范化**：内置 NFKC、Sequence、Prepend、Replace 等多种规范化器。
- **高级预分词**：支持 ByteLevel、Digits、Split 以及基于 Regex 的复杂切分（完美复刻 GPT-2/4 风格）。
- **高效轻量**：优化的 C++ 实现，依赖极少。
- **自包含**：内置经过深度裁剪的 Oniguruma 正则引擎，在保持强大 Unicode 支持的同时最小化体积。

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
- 支持 C++17 的编译器

```bash
mkdir build
cd build
cmake ..
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

## 文档

关于项目架构和技术实现的深度解析，请参阅技术文档 [doc/implementation_details_CN.md](doc/implementation_details_CN.md)。

## 许可证

Apache License 2.0。详情请见 [LICENSE](LICENSE) 文件。
