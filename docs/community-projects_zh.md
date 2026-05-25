# 社区项目

以下项目由社区构建和维护。感谢所有贡献者！注意，这些项目并不由 OmniVoice 团队官方支持。

如果你有想添加的项目，请提交 PR。

---

- **[ComfyUI-OmniVoice-TTS](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS)** —
  用于 OmniVoice 文本转语音生成的 ComfyUI 自定义节点。

- **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** —
  面向 omni-modality model（全模态模型）的高效模型推理框架。支持 OmniVoice 服务化。

- **[pyVideoTrans](https://github.com/jianchang512/pyvideotrans)** —
  带配音和字幕的视频翻译工具。支持将 OmniVoice 作为 TTS 引擎。

- **[MLX-Audio](https://github.com/Blaizzy/mlx-audio)** —
  基于 Apple MLX 框架构建的 TTS、STT 和 STS 库。支持 OmniVoice 以及其他模型，可在 Apple Silicon 上高效处理语音。

- **[RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)** —
  将文本实时转换为语音。支持 OmniVoice 作为 TTS 引擎。

- **[TTS-WebUI](https://github.com/rsxdalv/TTS-WebUI)** —
  支持多个 TTS 模型的 Gradio Web UI。支持将 OmniVoice 作为后端之一。

- **[OmniVoice-Studio](https://github.com/debpalash/OmniVoice-Studio)** —
  用于 OmniVoice 语音生成的桌面应用。

- **[omnivoice-server](https://github.com/maemreyo/omnivoice-server)** —
  用于通过 `/v1/audio/speech` 提供 OmniVoice 服务的 OpenAI 兼容 HTTP server（HTTP 服务器）。
  支持 voice profile（声音档案）以便持久化克隆、句子级 streaming（流式输出）以及可选 Bearer auth（Bearer 认证）。

- **[omnivoice-rs](https://github.com/FerrisMind/omnivoice-rs)** —
  GPU-first（GPU 优先）的 Rust workspace（工作区），用于 OmniVoice 推理、等价性校验、CLI 执行，以及基于 Candle 构建的 OpenAI 兼容 HTTP server。

- **[omnivoice-trtllm](https://github.com/tlitech/omnivoice-trtllm)** —
  使用 TensorRT-LLM 和 Triton Inference Server 在 Modal 上部署 OmniVoice TTS 模型，比 PyTorch 更快。

  - **[Auris](https://github.com/nikhilprasanth/Auris)** —
    离线有声书阅读器，支持 EPUB、PDF 和 TXT，使用本地 OmniVoice TTS，支持按角色区分声音和按书籍控制旁白。
