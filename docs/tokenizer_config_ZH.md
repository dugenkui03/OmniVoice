# OmniVoice tokenizer 配置说明

- `add_prefix_space: false`：分词前不在文本开头自动添加空格。
- `backend: "tokenizers"`：使用 Hugging Face `tokenizers` 作为分词后端。
- `bos_token: null`：没有设置通用的“序列开始”标记。
- `clean_up_tokenization_spaces: false`：解码文本时不自动清理空格。
- `eos_token: "<|im_end|>"`：使用 `<|im_end|>` 表示序列结束。
- `errors: "replace"`：遇到无法解码的内容时，用替代字符处理，而不是直接报错。
- `extra_special_tokens`：OmniVoice 自定义的控制标记：
  - `<|denoise|>`：去噪标记。
  - `<|lang_start|>`、`<|lang_end|>`：语言信息的开始和结束标记。
  - `<|instruct_start|>`、`<|instruct_end|>`：声音风格指令的开始和结束标记。
  - `<|text_start|>`、`<|text_end|>`：待合成文本的开始和结束标记。
- `is_local: true`：表示该 tokenizer 配置以本地可加载形式保存。
- `model_max_length: 131072`：允许的最大输入长度为 131072 个 token。
- `pad_token: "<|endoftext|>"`：批量处理不同长度文本时，使用 `<|endoftext|>` 补齐。
- `split_special_tokens: false`：特殊标记作为完整 token，不继续拆分。
- `tokenizer_class: "Qwen2Tokenizer"`：实际使用 Qwen2Tokenizer 处理文本。
- `unk_token: null`：没有单独设置“未知内容”标记。
