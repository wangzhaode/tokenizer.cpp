# 实现细节 (Implementation Details)

本文档概述了 `tokenizer.cpp` 的内部架构和关键实现决策。

## 架构

分词器遵循标准的 HuggingFace 分词流水线 (Tokenizer Pipeline)：

1.  **预分词 (Pre-tokenization)**:
    *   **Regex Split (`SplitPreTokenizer`)**:
        *   使用裁剪版 `Oniguruma` 引擎进行基于正则的切分。
        *   支持 GPT-2/4 风格的正则 (如 `'s|'t|...`)。
        *   处理特殊的 Unicode 属性正则 (如 `\p{L}+` 或 `\p{N}` )。
        *   支持 `invert` 行为 (保留/不保留分隔符)。
    *   **ByteLevel (`ByteLevelPreTokenizer`)**:
        *   将文本转换为字节序列，通常用于 GPT-2 风格的模型。
        *   处理 `use_regex` 标志，决定是否先进行正则切分。
    *   **Digits (`DigitsPreTokenizer` / `Digits`)**:
        *   处理数字切分逻辑 (如 Llama 3 将数字单独切分)。

2.  **规范化 (Normalization)**:
    *   **顺序组合 (`SequenceNormalizer`)**: 支持按顺序执行多个规范化步骤。
    *   **NFKC (`NFKCNormalizer`)**: 使用 `utf8proc` 库进行 Unicode NFKC 标准化。
    *   **替换 (`ReplaceNormalizer`)**: 基于正则或字符串的替换逻辑 (支持 `prepend` 行为)。
    *   **前缀 (`PrependNormalizer`)**: 添加特定前缀 (如 Llama 的 `_`)。

3.  **模型核心 (Model)**:
    *   **BPE (`BPE` / `BPEModel`)**:
        *   实现标准的字节对编码算法。
        *   支持 `dropout` (主要用于训练，推理时通常为 0)。
        *   **Rank 合并**: 使用预加载的 `merges` 表和 `ranks` 映射进行高效合并，优先级 (Rank) 越小越先合并。
        *   **Cache**: 包含 `cache` 机制加速常见单词的分词 (尽管在 C++ 实现中通常直接计算也足够快)。
    *   **WordPiece (`WordPiece` / `WordPieceModel`)**:
        *   支持 BERT 风格的最长匹配算法 (`max_input_chars_per_word`, `unk_token`)。
    *   **Unigram (`Unigram` / `UnigramModel`)**:
        *   基于概率的 Unigram 分词算法 (主要用于 AlBERT, SentencePiece 模型)。

4.  **解码 (Decoder)**:
    *   **ByteLevel (`ByteLevelDecoder`)**: 将字节级 token 还原为 UTF-8 字符串。
    *   **Fuse (`FuseDecoder`)**: 简单的 token 拼接。
    *   **Strip (`StripDecoder`)**: 处理 `content` 的 `lstrip`/`rstrip`，这在 Chat 模板渲染后处理中非常关键，用于去除特殊 tag 周围不必要的空格。
    *   **Replace (`ReplaceDecoder`)**: 逆向替换。

## 关键实现特性

### 1. Oniguruma 的深度裁剪
为了支持复杂的 Unicode 正则 (如 `\p{L}`) 同时保持轻量级：
*   我们内置了 `third_party/oniguruma`。
*   **裁剪**: 移除了所有非 UTF-8 编码支持 (EUC-JP, SJIS 等)，移除了 POSIX/GNU 兼容层，仅保留核心正则引擎。
*   **体积优化**: 使得最终二进制体积增加极小，远小于引入 ICU 或完整版 Oniguruma。

### 2. JSON 加载与兼容性
*   直接解析 HuggingFace 标准的 `tokenizer.json`。
*   使用 `nlohmann::json` 处理复杂的嵌套配置 (如 `normalizer` 中套 `Sequence` 再套 `Replace`)。
*   针对 `pre_tokenizer` 和 `normalizer` 的多态类型实现了工厂模式加载。

### 3. Unicode 处理
*   使用 `utf8proc` 进行字符迭代和宽字符属性查询。
*   `OnigRegex` 类封装了对 UTF-8 字符串的正则搜索，自动处理 `char*` 与 `OnigUChar*` 的转换。

### 4. 性能优化
*   **Token ID 映射**: 使用 `std::unordered_map` (或高效的 flat map) 存储词表。
*   **字符串视图**: 在内部处理中尽量减少字符串拷贝 (尽管为了接口安全，公共 API 接受 `std::string`).

## 测试策略

### "对齐测试" (Alignment Testing)

我们采用 **Golden Data** 对齐策略，而非仅单元测试：

1.  **Golden Data 生成 (`tests/generate_test_data.py`)**:
    *   使用 Python `transformers` 库加载目标模型。
    *   输入包含多语言、特殊符号、Emoji、代码片段的复杂文本。
    *   导出 `ids` (Encode 结果) 和 `decoded` (Decode 结果)。
2.  **C++ 验证**:
    *   C++ 测试程序加载相同的 `tokenizer.json`。
    *   执行 Encode/Decode。
    *   **断言**: `cpp_ids == py_ids` 且 `cpp_decoded == py_decoded`。

### 已通过验证的模型

(测试结果基于 `tests/model_tests.cpp` 的自动化运行)

*   **Llama**: `Llama-2-7b`, `Meta-Llama-3-8B-Instruct`, `Llama-3.2-3B-Instruct`
*   **Qwen**: `Qwen2.5-3B-Instruct`, `Qwen2.5-Coder-32B`, `Qwen/QwQ-32B`
*   **DeepSeek**: `DeepSeek-V3`, `DeepSeek-R1-Distill-Llama-8B`
*   **Phi**: `Phi-3.5-mini-instruct`
*   **Mistral**: `Ministral-3-3B-Instruct`
*   **GLM**: `GLM-4-9b-chat`
*   **Gemma**: `gemma-3-4b-it`