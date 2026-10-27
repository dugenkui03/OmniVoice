# 提示和注意事项

- **`ref_audio` 和 `instruct` 的组合**：
  当同时提供 `ref_audio` 和 `instruct` 且二者**冲突**时，模型大概率会跟随参考音频的风格。当二者**一致**时，`instruct` 可以提升其描述属性的克隆稳定性。一个典型例子是**中文方言克隆**：同时提供方言参考音频和匹配的方言 instruct（例如 `ref_audio="sichuan.wav", instruct="四川话"`），方言输出会更稳定。

- **短音频生成**：
  在没有参考音频时，模型可能无法稳定生成很短的音频片段（例如 1-2 秒）。如果需要生成短片段，建议向模型提供参考音频。

- **闽南语输入格式**：
  Min Nan Chinese（闽南语，也称 Hokkien）目前只能使用 [Tai-lo romanization](https://en.wikipedia.org/wiki/T%C3%A2i-l%C3%B4) 作为输入；当前模型版本不支持用汉字合成闽南语。
