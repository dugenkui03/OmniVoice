# AGENTS.md instructions for /Users/dugenkui/workspace/OmniVoice

## Books Writing

- 优化 `books/` 下的书稿前，先阅读 `books/AGENT_HANDOFF.md` 和 `books/chapter00_书籍介绍中.md`，明确本书定位、章节职责和读者导览。
- `books/` 下的内容应写成面向所有读者的科普书稿，而不是记录用户和 agent 的对话过程。
- 章节正文要直接讲概念、原理、模块职责和工程链路，避免出现“我们刚才讨论”“你说得对”“为什么不这么写”“可以这样区分”这类对话式或答疑式痕迹。
- 章节正文不要解释写作选择或回应式澄清，例如“本书关注的不是……而是……”“这里故意……”“读这一章时……”“不要只问……”“这张表不是为了……”“对第 X 章来说……”。应改成直接的技术陈述，例如“该链路通常分为……”“该模块位于……”“阅读系统方案时，可以按……定位模块边界”。
- 如果需要承接读者可能有的疑问，应改写成书稿表达，例如“常见误区是……”“工程上可以理解为……”“在完整链路中，这个模块负责……”。
- 短概念补充可以使用小贴片提示格式，优先用 Markdown 引用块：`> 💡 **小科普：概念名是什么？**`，下一段用 2-4 句解释定义、作用和所在链路。小贴片适合解释 loss、optimizer、latent、tokenizer 等读者可能临时卡住但又不适合打断主线的概念。
- 书稿中应适度使用加粗来突出关键阶段、关键结论和容易混淆的角色，例如 **训练阶段**、**推理阶段**、**主生成模型**、**波形还原模块**。如果 Markdown 渲染环境支持 HTML，可以少量使用 `<span style="color:#...">...</span>` 给关键术语做颜色强调，但不要让正文变得花哨。
- 解释流程、架构或模块关系时，优先使用模块化架构流程图 / 分组架构流程图：用节点表示处理模块，用箭头表示数据流；同一职能域用虚线分组框包起来并加组标题；核心模块用不同颜色高亮，输出结果用另一种颜色强调。
- Mermaid 图类型要服务于表达目标：如果重点是模块归属、数据流、架构位置，优先用 `flowchart LR`；如果重点是时间顺序、多模块协作、训练 / 推理交互，优先用 `sequenceDiagram`。
- Mermaid 图如果使用特殊颜色或虚线分组，应优先把轻量图例放进图中：`flowchart` 用一个靠边的小节点即可，例如 `图例：紫=数据｜橙=模块｜绿=输出`，不要用大块 `subgraph legend` 喧宾夺主；`sequenceDiagram` 可用简短 `Note over ...`。如果图内说明会显著变乱，再在图后增加一句简短图例。
- 画流程图时必须区分“数据 / 表示”和“处理单元 / 模型模块”：例如 waveform、mel、codec token、latent 是数据或中间表示；encoder、decoder、tokenizer、generator、vocoder 是处理单元。图中应通过节点文案、颜色或图例明确区分，避免写成 `waveform -> encoder -> token -> decoder` 但不说明哪些是结果、哪些是模块。
- 讲 TTS / AI 技术路线时，优先讲当前主流模型仍然常见、能帮助读懂 OmniVoice / IndexTTS / VoxCPM / CosyVoice 等系统的机制；RNN、CNN、Tacotron 等早期路线只作为必要背景，不与当前主流机制等量展开。
- 举例时优先使用用户关注的模型族：OmniVoice、IndexTTS / IndexTTS2、VoxCPM / VoxCPM2、CosyVoice / CosyVoice3；FastSpeech / FastSpeech 2 只在解释显式 duration 等历史基础机制时作为背景例子，避免反复作为主线代表。
- 第九章讲波形还原模块时，优先按“主生成对象 -> 还原模块 -> 输入输出 -> 推理期可干预点”组织内容；vocoder、codec decoder、AudioVAE decoder 是并列的还原路线，方案例子优先围绕 OmniVoice、IndexTTS2、VoxCPM2、CosyVoice3 展开，避免泛泛罗列旧 vocoder 年表。
- 解释相关概念时要区分层级：alignment（对齐机制）、Transformer / LLM backbone（主干网络）、diffusion / flow matching（生成范式）、codec token / mel / latent（生成空间）不是同一维度，不能混成一个分类。
- 如果某个小节讲的是总体架构中的某一段，优先在小节开头放一个 left-to-right 的 Mermaid 定位图：其他模块折叠成概览节点，当前小节负责的模块展开成子模块，帮助读者知道当前位置。
- 小节定位图必须清楚标出层级或阶段，例如“前端表示 / 条件输入”“模型内部”“模型输出 / 声学表示”“波形还原”。如果本节讲的是多种方案或多种声学表示，必须画成并列分支，不要画成单一路径或把并列概念压进一个节点里。
