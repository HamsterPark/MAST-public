# 各 provider 的视觉(图像输入)支持

抓取于 2026-08-11。Moonshot 的细节单独在 `moonshot_kimi_vision.md`(那家是我们的默认模型,
值得一整份);本文是**跨 provider 的能力对照**,对应代码里的单一真源
`config.model_supports_vision()` / `config._VISION_MODELS`。

> 这份表的用途只有一个:**让能力表里的每一行都有出处**。
> 表里写 ❌ 的,要能说清楚是「查过、确实不支持」还是「查不到」——这两件事不一样。

---

## 对照表

| Provider | 我们在用的 model id | 视觉 | 依据 |
|---|---|:--:|---|
| Moonshot / Kimi | `kimi-k3`, `kimi-k2.6`, `kimi-k2.7-code` | ✅ | 官方视觉模型列表逐字列出(见 `moonshot_kimi_vision.md`) |
| Moonshot / Kimi | `moonshot-v1-128k` | ❌ | **不在**官方列表里(带 `-vision-preview` 后缀的才在) |
| Anthropic | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` | ✅ | Messages API 原生支持图像;Models API 有 `capabilities.image_input` |
| MiniMax | `MiniMax-M3` | ✅ | [minimax_anthropic_compat.md](minimax_anthropic_compat.md) §已审计:Anthropic 兼容 `type="image"`,URL 或 base64 |
| MiniMax | `MiniMax-M2.7 / M2.5 / M2.1 / M2` | ❌ | 同上文档:仅文本 + 工具,**不支持图片和视频** |
| DeepSeek | `deepseek-v4-pro` | ❌ | 官方模型/价格文档**没有**图像请求格式,也没有公开的视觉模型 id |
| 智谱 GLM | `glm-5.2`(及 `glm-5.1`) | ❌ | docs.bigmodel.cn 的 GLM-5.2 页面**输入模态**一栏写的是「文本」 |
| Qwen / DashScope | `qwen3.7-max` | ❓ **查不到** | 这个 id **根本不在**阿里云当前模型列表上(现网是 `qwen3.8-max`) |

代码里 ✅ 的进 `_VISION_MODELS`;❌ 和 ❓ 都不进(默认 False = 纯文本降级)。

---

## 逐条说明

### Anthropic ✅

所有在产 Claude 模型都收图像。**原生块形状与 OpenAI 完全不同**:

```json
{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
```

我们发的是 OpenAI 形状的 `image_url` + data URI,靠 `langchain_anthropic` 在序列化时转换成上面这个。
**这是一个我们不拥有的第三方行为**,所以钉了测试
(`tests/v2/agents/_shared/test_vision_channel.py::test_block_also_converts_on_the_anthropic_path`)——
哪天它不转了,MiniMax 和 Claude 会开始收到读不懂的块,而我们这边一行代码都没改过。

同一条转换也是 **MiniMax 能共用一份发射代码的原因**(MiniMax 走 ChatAnthropic)。

### MiniMax ✅(仅 M3)

本仓**早就审计过**,在 [minimax_anthropic_compat.md](minimax_anthropic_compat.md):M3 支持 `type="image"`,URL 或 base64,
JPEG/PNG/GIF/WEBP,单图 ≤ 10 MB,请求体 ≤ 64 MB;M2.x 全系不支持。
`config.py:214` 那个零消费者的 `MINIMAX_VISION_MODELS` 存根记的就是这一条事实——
它被吸收进 `_VISION_MODELS`,事实没丢,只是不再有第二个地方回答同一个问题。

### DeepSeek ❌

DeepSeek 的官方模型与价格文档列的是思考模式、JSON 输出、工具调用、前缀续写、FIM 补全
这些**文本**能力,**没有**图像请求格式,也没有公开的视觉模型 id;发 `image_url` 会在解析阶段
就被拒掉,轮不到模型。

⚠️ 这条是「官方文档里没有」而不是「官方明确说不支持」——但对我们的判断没有区别:
没有文档化的请求格式,就没有可以发的东西。

### 智谱 GLM ❌ —— 这条差点写反,记一笔

搜索引擎的摘要**信誓旦旦地说** GLM-5.2「原生处理图片、视频等多模态输入」,还给出了
`{"type":"image_url","url":"data:image/png;base64,..."}` 的用法。

去翻官方文档页(docs.bigmodel.cn 的 GLM-5.2)才发现:**输入模态一栏写的是「文本」**,
整页没有出现图片/图像/vision,所有请求示例都只有文本,也没有 glm-5v 之类的视觉变体。

摘要把 GLM-5 的宣传口径、别的模型的用法混成了一段看起来很合理的话。
**这正是本次任务点名要避开的形状**:一个听起来对、格式也像模像样的答案,不是事实。
按摘要写就会给 GLM 打开视觉开关,结果是操作员一换 GLM 就每张图 400。

### Qwen ❓ 查不到(而且是最容易看错的一种「查不到」)

`qwen3.7-max` —— 我们代码里的那个 id —— **不在**阿里云当前的模型列表上;
现网这条线是 `qwen3.8-max` / `qwen3.7-plus`。而 `qwen3.8-max` 和 `qwen3.7-plus`
确实同时出现在「文本生成」和「图像与视频 · 理解」两栏里。

**但那是别的 model id。** 「这个家族支持视觉」推不出「我们用的这个 id 支持视觉」——
Moonshot 那对 `moonshot-v1-128k` / `moonshot-v1-128k-vision-preview` 已经把这件事
演示得很清楚了:同前缀、同家族,相反的答案。

所以 Qwen 这一格是**诚实的空白**,不是 ❌ 也不是 ✅。要填上它,需要先确认
`qwen3.7-max` 这个 id 现在到底还在不在、对应哪个模型(顺带一提:这也值得单独查一次,
因为我们可能正在往一个已经改名的 id 上发请求)。

⚠️ **而且 Qwen 这条线上「快照 id 与基础 id 能力不同」是有报告的**:另一路独立调查
(2026-08-11)称 `qwen3.7-max` 是文本、而某个 `qwen3.7-max-<日期>` 快照支持图像。
我没能在官方页面上核实那个具体快照,所以不写进表里 —— 但它把风险说清楚了:
我们 `config.py` 里**恰好就有**一个快照预设 `qwen3.7-max-preview` →
`qwen3.7-max-2026-05-20`(又是第三个日期)。

⇒ 结论不变(两个 Qwen id 都不进 `_VISION_MODELS`),但**谁要开 Qwen 视觉,必须逐个
快照 id 去官方列表上核**,不能从「qwen3.7 系支持图像」推到我们在用的那两个 id 上。
这与 Moonshot 的 `moonshot-v1-128k` / `-vision-preview` 是同一个陷阱,
区别只是那边没咬中在用 id(七个 agent 全 pin `kimi-k3`),这边可能咬中。

---

## 加一个 provider 时该做什么

1. 找到**官方**文档里那句话(不是搜索摘要、不是第三方博客、不是另一家的类比);
2. 确认它讲的是**我们在用的那个 model id**,不是同家族的另一个;
3. ✅ 才进 `config._VISION_MODELS`;查不到就留空白并在本文写下「查不到」;
4. 块格式若与 OpenAI 的 `image_url` 不同,确认序列化层会转换,**并为那个转换加测试**
   (它不是我们的代码,不会因为我们的改动而失效,只会因为别人的改动而失效)。
