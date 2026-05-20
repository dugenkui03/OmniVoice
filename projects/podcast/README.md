# Podcast Project

这个目录用于把 OmniVoice 当前项目的 TTS 能力封装成一个“播客生成”小工程。第一版目标是用 `/Users/dugenkui/workspace/ai_dubbing` 中已经切好的《甄嬛传》E11 下素材，生成一段“甄嬛版”中文学习播客。

## 当前链路

```mermaid
flowchart LR
    A["ai_dubbing E11 下分析 JSON"] --> C["播客脚本"]
    B["原始台词 line_0000.wav..."] --> D["角色声音参考音频"]
    D --> E["OmniVoice voice clone prompt"]
    C --> F["逐句 TTS"]
    E --> F
    F --> G["分段 WAV"]
    G --> H["拼接整期播客 WAV/MP3"]
    C --> I["脚本与 timing 报告"]
```

## 运行方式

先在 OmniVoice 仓库根目录执行：

```bash
uv run python projects/podcast/generate_podcast.py
```

默认会：

- 读取本地模型：`.models/OmniVoice`
- 读取素材：`/Users/dugenkui/workspace/ai_dubbing/test_data/测试输出/.../甄嬛传_E11_下集_教科书级向上管理_曹贵人巧言化解华妃雷霆之怒`
- 输出到：`output/podcast/zhenhuan_e11_down_<时间戳>/`

快速冒烟测试可以只跑前 3 句：

```bash
uv run python projects/podcast/generate_podcast.py --max-lines 3
```

只生成参考素材、脚本和报告，不跑 TTS：

```bash
uv run python projects/podcast/generate_podcast.py --skip-tts
```

## 输出内容

- `refs/*.wav`：用原剧台词拼出来的角色克隆参考音频。
- `segments/*.wav`：每一句播客台词生成出的 TTS 片段。
- `zhenhuan_e11_down_podcast.wav`：拼接后的整期播客。
- `zhenhuan_e11_down_podcast.mp3`：如果本机有 `ffmpeg`，会额外导出 MP3。
- `podcast_script.md`：可读版播客脚本。
- `podcast_lines.json`：结构化播客脚本，记录说话人、参考音频、时长等。
- `timing.tsv`：每句生成耗时、音频时长、RTF 等。

## 设计说明

这里没有改 OmniVoice 主项目代码，而是在 `projects/podcast` 中做项目化封装。这样后面可以继续加：

- 不同剧集的脚本模板。
- 不同角色的参考音频选择策略。
- 更细的情绪控制和非语言标签，如 `[laughter]`、`[sigh]`。
- 自动从 `episode_analyses/*.json` 里生成播客大纲。
