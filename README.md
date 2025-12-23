# tokenizer.cpp

![License](https://img.shields.io/badge/license-Apache%20License%202.0-green)
![Build Status](https://github.com/wangzhaode/tokenizer.cpp/actions/workflows/build.yml/badge.svg)
[![中文版本](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-green)](README_CN.md)

A production-ready, highly compatible C++ tokenizer library widely compatible with HuggingFace models.

It provides a high-performance C++ implementation for modern LLM tokenization pipelines, supporting BPE, WordPiece, and Unigram models with full compatibility for complex normalizations and pre-tokenization rules.

## Features

- **HuggingFace Compatible**: Loads directly from `tokenizer.json`.
- **Comprehensive Support**: Supports BPE, WordPiece, and Unigram models.
- **Complex Normalization**: Implements NFKC, Sequence, Prepend, Replace, and more.
- **Advanced Pre-tokenization**: Supports ByteLevel, Digits, Split, and Regex-based patterns (GPT-2/4 style).
- **Efficient**: Optimized C++ implementation using minimal dependencies.
- **Self-Contained**: Includes pruned versions of optimizations like Oniguruma for minimal footprint.

## Supported Models

Verified and fully compatible with tokens from:
- **Llama 2 / 3 / 3.1 / 3.2**
- **Qwen 2.5 / 3 / 2.5-Coder**
- **DeepSeek V2 / V3 / R1**
- **Phi-3.5**
- **GLM-4**
- **Mistral**
- **Gemma**
- And many others...

## Integration

The library allows easy loading and usage of tokenizers.

### Build Instructions

**Prerequisites**:
- CMake 3.10+
- C++17 compatible compiler

```bash
mkdir build
cd build
cmake ..
make
```

### Run Tests

The repository includes a comprehensive test suite validating against HuggingFace `tokenizers` output for various models.

```bash
./test_main
```

## Usage

### Basic Tokenization

```cpp
#include "tokenizer.hpp"
#include <iostream>

int main() {
    // Load tokenizer from HuggingFace-style tokenizer
    auto tokenizer = tokenizer::AutoTokenizer::from_pretrained("path/to/tokenizer/dir");

    std::string prompt = "Hello, world!";

    // Encode
    std::vector<int> ids = tokenizer->encode(prompt);

    // Decode
    std::string decoded = tokenizer->decode(ids);

    std::cout << "Encoded IDs: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << "\nDecoded: " << decoded << std::endl;

    return 0;
}
```

## Documentation

For deep technical details on the implementation and architecture, see [doc/implementation_details_CN.md](doc/implementation_details_CN.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) file for details.
