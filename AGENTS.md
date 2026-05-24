# AGENTS.md instructions for /Users/dugenkui/workspace/OmniVoice

## Books Writing

- 优化 `books/` 下的书稿前，先阅读 `books/AGENT_HANDOFF.md` 和 `books/chapter00_书籍介绍中.md`，明确本书定位、章节职责和读者导览。
- `books/` 下的内容应写成面向所有读者的科普书稿，而不是记录用户和 agent 的对话过程。
- 章节正文要直接讲概念、原理、模块职责和工程链路，避免出现“我们刚才讨论”“你说得对”“为什么不这么写”“可以这样区分”这类对话式或答疑式痕迹。
- 如果需要承接读者可能有的疑问，应改写成书稿表达，例如“常见误区是……”“工程上可以理解为……”“在完整链路中，这个模块负责……”。
- 解释流程、架构或模块关系时，优先使用模块化架构流程图 / 分组架构流程图：用节点表示处理模块，用箭头表示数据流；同一职能域用虚线分组框包起来并加组标题；核心模块用不同颜色高亮，输出结果用另一种颜色强调。
- 讲 TTS / AI 技术路线时，优先讲当前主流模型仍然常见、能帮助读懂 OmniVoice / IndexTTS / VoxCPM / CosyVoice 等系统的机制；RNN、CNN、Tacotron 等早期路线只作为必要背景，不与当前主流机制等量展开。
- 举例时优先使用用户关注的模型族：OmniVoice、IndexTTS / IndexTTS2、VoxCPM / VoxCPM2、CosyVoice / CosyVoice3；FastSpeech / FastSpeech 2 只在解释显式 duration 等历史基础机制时作为背景例子，避免反复作为主线代表。
- 第九章讲 vocoder / decoder / 波形还原时，优先按“主生成对象 -> 还原模块 -> 输入输出 -> 推理期可干预点”组织内容；方案例子优先围绕 OmniVoice、IndexTTS2、VoxCPM2、CosyVoice3 展开，避免泛泛罗列旧 vocoder 年表。
- 解释相关概念时要区分层级：alignment（对齐机制）、Transformer / LLM backbone（主干网络）、diffusion / flow matching（生成范式）、codec token / mel / latent（生成空间）不是同一维度，不能混成一个分类。
- 如果某个小节讲的是总体架构中的某一段，优先在小节开头放一个 left-to-right 的 Mermaid 定位图：其他模块折叠成概览节点，当前小节负责的模块展开成子模块，帮助读者知道当前位置。
- 小节定位图必须清楚标出层级或阶段，例如“前端表示 / 条件输入”“模型内部”“模型输出 / 声学表示”“波形还原”。如果本节讲的是多种方案或多种声学表示，必须画成并列分支，不要画成单一路径或把并列概念压进一个节点里。
