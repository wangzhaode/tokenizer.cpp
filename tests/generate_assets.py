import os
import shutil
import json
from modelscope.hub.file_download import model_file_download
from transformers import AutoTokenizer

# ================= 配置区域 =================

SAVE_ROOT = "./models"

TARGET_MODELS = [
    # 1. BPE Tokenizer Models
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen2.5-7B-Instruct-1M",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    # "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen3Guard-Gen-4B",
    # "Qwen/Qwen3Guard-Stream-4B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-R1",
    "ZhipuAI/GLM-4.5V",
    "ZhipuAI/GLM-4.6V",
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    # "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "google/gemma-3-4b-it",
    "google/gemma-3n-E4B-it",
    "mistralai/Ministral-3-3B-Instruct-2512",
    "LLM-Research/llama-2-7b",
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    "LLM-Research/Llama-3.2-3B-Instruct",
    "LLM-Research/Phi-3.5-mini-instruct",
    "LLM-Research/Phi-3.5-vision-instruct",
    "LLM-Research/phi-4",
    "LLM-Research/Phi-4-mini-reasoning",
    # # 2. WordPiece Tokenizer Models
    # "google-bert/bert-base-uncased",
    # "google-bert/bert-base-multilingual-cased",
    # "AI-ModelScope/bge-large-zh",
    # "iic/gte_sentence-embedding_multilingual-base",
    # "iic/gte-multilingual-reranker-base",
    # # 3. Unigram Tokenzier Models
    # "AI-ModelScope/t5-small",
]

CONFIG_FILES = [
    "tokenizer_config.json",
]

# ================= 增强型测试语料库 =================
# 覆盖：基础英文、多语言、代码、Emoji、空白字符边界、Unicode规范化、Byte-level fallback

TEST_CORPUS = [
    # 1. 基础 & 标点
    "Hello World",
    "Hello  World", # 多空格
    " don't ",      # 前后空格 + 缩写
    "The quick brown fox jumps over the lazy dog.",

    # 2. 多语言 (CJK + 混合)
    "你好世界",
    "こんにちは世界", # 日语
    "안녕하세요",     # 韩语
    "I love 中国",    # 中英混合
    "早C晚A",        # 典型混合网络用语

    # 3. 代码 (关注缩进、换行、符号)
    "def main():\n    print('hello world')",
    "#include <iostream>\nusing namespace std;",
    "const a = 10; // comment",
    "print(a_b_c)",

    # 4. 数字与数学
    "1234567890",
    "3.14159",
    "x^2 + y_2 = z",

    # 5. Emoji & 特殊符号 (测试 Unigram/BPE 对 Byte 的处理)
    "😊 😂 🥺",
    "👨‍👩‍👧‍👦", # 组合 Emoji (ZWJ)
    "Hash#Tag $Price %Percent &And",

    # 6. 边界与空白 Case (Tokenizer 的噩梦)
    "",             # 空串
    "   ",          # 纯空格
    "\n",           # 纯换行
    "\t\n\r",       # 混合控制符
    "Hello\nWorld",

    # 7. URL & Email (通常不应被切分得太碎)
    "https://github.com/google/gemini",
    "user.name@example.com",

    # 8. 罕见字符 / Byte Fallback 测试
    # 某些 Tokenizer 遇到未识别字符会回退到 Byte 编码，或者输出 <unk>
    "Ã", "é", "β", "①",
]

# ================= Chat 模板测试语料 =================
# 用于测试 apply_chat_template 的功能
CHAT_TEST_CORPUS = [
    # ===== 1. 基础场景 =====
    {
        "name": "basic_user",
        "messages": [
            {"role": "user", "content": "Hi"}
        ]
    },
    {
        "name": "system_user_assistant",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant specialized in Python."},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am an AI assistant created to help you with Python programming."}
        ]
    },

    # ===== 2. 边界场景 =====
    {
        "name": "consecutive_users",
        "messages": [
            {"role": "user", "content": "Part 1: What is machine learning?"},
            {"role": "user", "content": "Part 2: Give me a simple example."}
        ]
    },
    {
        "name": "gen_prompt_true",
        "messages": [{"role": "user", "content": "Hello, please help me."}],
        "add_generation_prompt": True
    },
    {
        "name": "gen_prompt_false",
        "messages": [{"role": "user", "content": "Hello, please help me."}],
        "add_generation_prompt": False
    },

    # ===== 3. 多轮复杂对话 =====
    {
        "name": "multi_turn_code",
        "messages": [
            {"role": "system", "content": "You are a senior software engineer."},
            {"role": "user", "content": "How do I reverse a string in Python?"},
            {"role": "assistant", "content": "You can use slicing: `s[::-1]` or `''.join(reversed(s))`"},
            {"role": "user", "content": "What about for a list?"},
            {"role": "assistant", "content": "For lists: `lst[::-1]`, `list(reversed(lst))`, or `lst.reverse()` (in-place)"},
            {"role": "user", "content": "Which one is fastest?"}
        ],
        "add_generation_prompt": True
    },

    # ===== 4. 工具调用场景 (Tool Calls) =====
    {
        "name": "tool_call_weather",
        "messages": [
            {"role": "user", "content": "What's the weather like in New York?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"New York\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "name": "get_weather",
                "content": "{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}"
            }
        ],
        "add_generation_prompt": True
    },
    {
        "name": "parallel_tool_calls",
        "messages": [
            {"role": "user", "content": "Compare weather in Tokyo and London"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_tokyo",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}
                    },
                    {
                        "id": "call_london",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_tokyo",
                "name": "get_weather",
                "content": "{\"temp\": 28, \"condition\": \"humid\"}"
            },
            {
                "role": "tool",
                "tool_call_id": "call_london",
                "name": "get_weather",
                "content": "{\"temp\": 15, \"condition\": \"rainy\"}"
            }
        ],
        "add_generation_prompt": True
    },

    # ===== 5. 推理/思维链场景 =====
    {
        "name": "reasoning_content",
        "messages": [
            {"role": "user", "content": "Solve: If a train travels 120km in 2 hours, what's the speed?"},
            {
                "role": "assistant",
                "content": "The speed is 60 km/h.",
                "reasoning_content": "Let me think step by step:\n1. Distance = 120 km\n2. Time = 2 hours\n3. Speed = Distance / Time = 120 / 2 = 60 km/h"
            }
        ]
    },

    # ===== 6. 多语言与特殊字符 =====
    {
        "name": "multilingual_complex",
        "messages": [
            {"role": "system", "content": "你是一个多语言AI助手。You can speak multiple languages."},
            {"role": "user", "content": "Translate 'Hello World' to: 中文、日本語、한국어"},
            {"role": "assistant", "content": "Here are the translations:\n- 中文: 你好世界\n- 日本語: こんにちは世界\n- 한국어: 안녕하세요 세계"}
        ]
    },
    {
        "name": "escape_characters",
        "messages": [
            {"role": "user", "content": "Explain these escape sequences: \\n \\t \\r \\\\ \\\""},
            {"role": "assistant", "content": "Here's what each means:\n- \\n = newline\n- \\t = tab\n- \\r = carriage return\n- \\\\ = backslash\n- \\\" = double quote"}
        ]
    },
    {
        "name": "code_with_special_chars",
        "messages": [
            {"role": "user", "content": "Write a regex to match email addresses"},
            {"role": "assistant", "content": "```python\nimport re\npattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\nemail = 'user@example.com'\nif re.match(pattern, email):\n    print('Valid!')\n```"}
        ]
    },

    # ===== 7. 长文本与 Markdown =====
    {
        "name": "markdown_formatting",
        "messages": [
            {"role": "user", "content": "Show me markdown formatting examples"},
            {"role": "assistant", "content": "# Heading 1\n## Heading 2\n\n**Bold** and *italic* text.\n\n- Bullet point 1\n- Bullet point 2\n\n```python\ndef hello():\n    return 'world'\n```\n\n| Col1 | Col2 |\n|------|------|\n| A    | B    |\n|------|------|\n| C    | D    |"}
        ]
    },

    # ===== 8. 空内容与边界 =====
    {
        "name": "empty_assistant_content",
        "messages": [
            {"role": "user", "content": "Say nothing"},
            {"role": "assistant", "content": ""}
        ]
    },
    {
        "name": "whitespace_only",
        "messages": [
            {"role": "user", "content": "   \n\t   "}
        ]
    },

    # ===== 9. Emoji 与 Unicode =====
    {
        "name": "emoji_conversation",
        "messages": [
            {"role": "user", "content": "Respond with emojis only: how are you?"},
            {"role": "assistant", "content": "😊👍✨🎉"}
        ]
    },
    {
        "name": "unicode_math_symbols",
        "messages": [
            {"role": "user", "content": "Write the quadratic formula using Unicode"},
            {"role": "assistant", "content": "x = (-b ± √(b² - 4ac)) / 2a\n\nOr in fancy form:\n𝑥 = (−𝑏 ± √(𝑏² − 4𝑎𝑐)) / 2𝑎"}
        ]
    },

    # ===== 10. 日期注入模拟 =====
    {
        "name": "date_injection",
        "messages": [
            {"role": "system", "content": "Current Date: 2025-12-17. You are a calendar assistant."},
            {"role": "user", "content": "What's today's date?"},
            {"role": "assistant", "content": "Today is December 17, 2025."}
        ]
    },

    # ===== 11. JSON 内容 =====
    {
        "name": "json_in_content",
        "messages": [
            {"role": "user", "content": "Parse this JSON: {\"name\": \"John\", \"age\": 30, \"skills\": [\"python\", \"javascript\"]}"},
            {"role": "assistant", "content": "The JSON contains:\n- name: John\n- age: 30\n- skills: python, javascript"}
        ]
    },

    # ===== 12. 超长系统提示 =====
    {
        "name": "long_system_prompt",
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant with the following capabilities:\n1. Code review and debugging\n2. Algorithm design and optimization\n3. System architecture consultation\n4. Technical documentation writing\n5. Best practices recommendation\n\nRules:\n- Always provide detailed explanations\n- Include code examples when relevant\n- Consider edge cases\n- Follow security best practices"},
            {"role": "user", "content": "Review my code"}
        ],
        "add_generation_prompt": True
    },
]

# ================= 功能函数 =================

def generate_test_cases(tokenizer, output_dir):
    """
    使用加载好的 tokenizer 生成测试用例
    包含:
    1. basic 类型: 基础 tokenization 测试
    2. chat  类型: apply_chat_template 测试
    """
    cases_path = os.path.join(output_dir, "test_cases.jsonl")
    print(f"  🧪 生成测试用例 -> {cases_path}")

    with open(cases_path, "w", encoding="utf-8") as f:
        # ===== 1. 基础 Tokenization 测试 =====
        for text in TEST_CORPUS:
            try:
                # 1. 纯分词模式 (add_special_tokens=False)
                enc_raw = tokenizer(text, add_special_tokens=False)
                tokens_raw = tokenizer.convert_ids_to_tokens(enc_raw["input_ids"])

                # 2. 完整模式 (add_special_tokens=True)
                enc_full = tokenizer(text, add_special_tokens=True)

                record = {
                    "type": "basic",
                    "input": text,
                    "ids_raw": enc_raw["input_ids"],
                    "tokens_raw": tokens_raw,
                    "ids_full": enc_full["input_ids"],
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"    ⚠️ Basic Case Error '{text}': {e}")

        # ===== 2. Chat Template 测试 =====
        # 检查 tokenizer 是否支持 chat_template
        if not hasattr(tokenizer, 'apply_chat_template') or tokenizer.chat_template is None:
            print(f"    ⚠️ Tokenizer 不支持 chat_template，跳过 chat 测试")
            return

        for chat_case in CHAT_TEST_CORPUS:
            try:
                messages = chat_case["messages"]
                add_generation_prompt = chat_case.get("add_generation_prompt", False)

                # 使用 apply_chat_template 生成格式化文本
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )

                # 对格式化文本进行 tokenization
                enc_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    return_tensors=None  # 返回 list 而非 tensor
                )

                record = {
                    "type": "chat",
                    "name": chat_case["name"],
                    "messages": messages,
                    "add_generation_prompt": add_generation_prompt,
                    "formatted_text": formatted_text,
                    "ids": enc_ids,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"    ⚠️ Chat Case Error '{chat_case['name']}': {e}")

def process_tokenizers():
    print(f"🚀 开始处理 Tokenizer 数据与测试集 (共 {len(TARGET_MODELS)} 个模型)\n")

    for idx, model_id in enumerate(TARGET_MODELS, 1):
        folder_name = model_id.split("/")[-1]
        local_dir = os.path.join(SAVE_ROOT, folder_name)
        os.makedirs(local_dir, exist_ok=True)

        print(f"[{idx}/{len(TARGET_MODELS)}] 处理: {model_id}")

        # --- 步骤 1: 下载 Config 文件 ---
        for filename in CONFIG_FILES:
            try:
                cached_path = model_file_download(model_id=model_id, file_path=filename, revision='master')
                shutil.copy(cached_path, os.path.join(local_dir, filename))
            except Exception:
                pass

        # --- 步骤 2: 准备 tokenizer.json ---
        target_json_path = os.path.join(local_dir, "tokenizer.json")
        json_ready = False

        # 2.1 尝试直接下载
        try:
            cached_path = model_file_download(model_id=model_id, file_path="tokenizer.json", revision='master')
            shutil.copy(cached_path, target_json_path)
            # print(f"  ✅ [Download] tokenizer.json")
            json_ready = True
        except Exception:
            pass

        # 2.2 下载失败则转换 (针对 BERT 等只有 vocab.txt 的模型)
        if not json_ready:
            try:
                print(f"  🔄 [Convert] 正在从原始模型转换 Fast Tokenizer...")
                # trust_remote_code=True 允许执行模型仓库里的 Python 代码 (对 Qwen/GLM 必须)
                temp_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                if temp_tokenizer.is_fast:
                    temp_tokenizer.backend_tokenizer.save(target_json_path)
                    print(f"  ✅ [Saved] 已生成 tokenizer.json")
                    json_ready = True
                else:
                    print(f"  ❌ [Error] 模型不支持 Fast Tokenizer")
            except Exception as e:
                print(f"  ❌ [Error] 转换失败: {str(e)[:100]}")

        # --- 步骤 3: 重新加载本地模型并生成测试用例 ---
        if json_ready:
            try:
                # 关键：从【本地目录】加载 Tokenizer
                # 这样保证生成的测试数据与磁盘上的 tokenizer.json 绝对一致
                # 避免内存中的 tokenizer 与磁盘文件版本不一致的情况
                local_tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

                # 生成测试数据
                generate_test_cases(local_tokenizer, local_dir)

            except Exception as e:
                print(f"  ❌ [Error] 本地加载或生成测试失败: {e}")
        else:
            # 清理无效目录
            if os.path.exists(local_dir) and not os.listdir(local_dir):
                os.rmdir(local_dir)

        print("-" * 40)

    print(f"\n🎉 全部完成! 数据保存在: {os.path.abspath(SAVE_ROOT)}")

if __name__ == "__main__":
    process_tokenizers()