# Moonshot / Kimi — Vision(图像输入)

Source（官方,抓取于 2026-08-11）:
- https://platform.kimi.ai/docs/guide/use-kimi-vision-model （视觉模型使用指南）
- https://platform.kimi.ai/docs/api/estimate （estimate-token-count 端点)

> 建这份文档的原因:`agents/_shared/models.py:35-41` 把 KIMI_K3 注释成
> 「native vision」,但 `docs/api_providers/` 里**没有任何 Moonshot 视觉文档** ——
> 也就是说在此之前,"Kimi 能看图" 只是一句**代码注释**,不是审计过的事实。
> 本仓规矩是「改 provider 参数前先读 docs/api_providers/」,那就得先有得读。

---

## 1. 哪些模型支持图像输入

官方页面逐字列出的视觉模型 id:

```
kimi-k3
kimi-k2.5
kimi-k2.6
kimi-k2.7-code
kimi-k2.7-code-highspeed
moonshot-v1-8k-vision-preview
moonshot-v1-32k-vision-preview
moonshot-v1-128k-vision-preview
```

对 MAST 的结论:

| MAST 常量 | model id | 视觉 | 依据 |
|---|---|---|---|
| `KIMI_K3` | `kimi-k3` | ✅ 支持 | 官方列表首项。**这是全部 7 个 agent 的默认模型** |
| `KIMI_K2_6` | `kimi-k2.6` | ✅ 支持 | 官方列表 |
| `KIMI_K2_7_CODE` | `kimi-k2.7-code` | ✅ 支持 | 官方列表 |
| `MOONSHOT_128K` | `moonshot-v1-128k` | ❌ **不支持** | 官方列表里只有 `moonshot-v1-128k-**vision-preview**`,**不带后缀的那个不在列表里** |

⚠️ 最后一行是这份文档最容易被读漏、也最容易出错的一条:
`moonshot-v1-128k` 和 `moonshot-v1-128k-vision-preview` 是**两个不同的 model id**,
差一个后缀就从"能看图"变成"不能看图"。能力表里必须逐 id 写,不能按前缀 `moonshot-*` 一刀切。
(同一个 id 在 `config._MODEL_THINKING_OVERRIDE` 里也是唯一的 thinking 例外 —— 它是纯 chat 模型。)

## 2. content-block 的确切格式

官方 Python 示例(逐字):

```python
completion = client.chat.completions.create(
    model="kimi-k3",
    messages=[
        {"role": "system", "content": "You are Kimi."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe the content of the image.",
                },
            ],
        },
    ],
)
```

要点:

- 块类型是 **`image_url`**,里面是一个**对象** `{"url": ...}` —— 不是裸字符串。
- **没有 `detail` 字段**。官方页面通篇未出现 `detail`(OpenAI 有,Moonshot 没有)。
  不要因为"OpenAI 兼容端点"就顺手加 `detail:"high"` —— 那是从别家推断格式,正是本条要防的。
- 官方原文:**"When using a Vision model, `message.content` must be an `array[object]`
  (that is, a JSON array)."** 用视觉时 content 必须是数组,不能是字符串。

### URL 还是 base64?——**只能 base64**

官方原文:

> "URL-formatted images: Not supported, currently only supports base64-encoded
> image content and images/videos uploaded via file ID"

即:

- ❌ 远程 `https://...` 图片链接 **不支持**;
- ✅ **base64 data URI**:`data:image/{format};base64,{encoded_data}`;
- ✅ 或经文件上传拿到的 `ms://` file id。

字段名叫 `image_url` 但**放的不是 URL 而是 data URI** —— 这个命名会误导人,
写代码时别按字面理解去塞一个 http 链接。

对 MAST 是好消息:`webui/scan_preview.py:151 render_scan_thumbnail()` 的返回值本来就是
`"data:image/png;base64," + b64` (见该函数末尾),**即插即用**,不需要第二个渲染器,
也不需要额外拼前缀。

## 3. 图像 token 计费

**官方没有公布像素→token 的换算公式。** 这是实话,不是没查到。
官方给出的是一个**接口**,不是一个公式:

> "Images and videos use dynamic token calculation: you can obtain the token
> consumption of a request containing images or videos through the estimate
> tokens API before starting"
>
> "the higher the image resolution, the more tokens it consumes."

官方计量手段 —— `POST https://api.moonshot.ai/v1/tokenizers/estimate-token-count`:

- 请求体:`{"model": "<id>", "messages": [...]}`,`messages[].content` **接受
  `text` / `image_url` / `video_url` 块**(和 chat 请求同构);
- 响应体:`{"data": {"total_tokens": <int>}}`。

也就是说:**要报数就调这个接口,不要在代码里写一个自己编的公式。**

### 半官方数据点(标注来源等级,不要当规格用)

Moonshot 官方论坛 forum.moonshot.ai 上有一条 thread
(https://forum.moonshot.ai/t/.../450,抓取 2026-08-11),回帖者是 Moonshot 侧人员
(未见 staff 徽章,语气与技术细节像内部人)。**这是论坛回帖,不是文档**:

- 实测:3509×4963 px → 4,394 tokens;裁剪成 1654×4343 px → 4,396 tokens(几乎一样);
- 解释:超过分辨率上限的图**在网关侧先降采样**再 tokenize,
  "you are only charged for the final scaled size, not your original massive upload";
- 提到分块:"scaled dimensions are first divided by 14, and the result is rounded up
  to the nearest whole integer (ceiling)"。

⚠️ 上面那个 "÷14 向上取整" **不足以复现 4,394 这个数**(按 14 px 块算会得到约 1.7 万),
按 28 px 有效块算才对得上(≈4,290)。所以**我们不知道确切公式**,论坛这段只说明了量级和
"先降采样再计费"这件事。**不要把它写进代码当公式用。**

### 对 MAST 的量级结论(标注为估算)

一帧 512 px 缩略图,按上面两种读法给出区间:

| 读法 | 每帧 token(估算) | 占 kimi-k3 1M 上下文 |
|---|---|---|
| 28 px 有效块(与论坛实测对得上) | ⌈512/28⌉² = 19² ≈ **361** | 0.04 % |
| 14 px 块(论坛字面) | ⌈512/14⌉² = 37² ≈ **1,369** | 0.14 % |

**即使取悲观读法也只有千分之一点四的上下文**,对"给 agent 看一张扫描图"这个用途完全可忽略。
(先前审计给的 255–324 token/帧 是同一量级的另一个估算;三个数都不是官方数,
真要报账就调 estimate-token-count。)

## 4. 尺寸 / 张数 / 总量限制

官方原文:

> "We recommend that image resolution does not exceed 4k (4096×2160), and video
> resolution does not exceed FHD (1920×1080)."
>
> "Image quantity: The Vision model has no limit on the number of images, but
> ensure that the request body size does not exceed 100M."

- 分辨率:建议 ≤ 4K(4096×2160);超了不报错,**网关自动降采样**(见上)。
- 张数:**无上限**,但整个请求体 ≤ 100 MB。
- MIME:`image/jpeg`, `image/png`, `image/gif`, `image/webp`, `image/bmp`,
  `image/heic`, `image/heif`。

我们发的是 512 px PNG(约 74 KB / 98.6 K base64 字符),三条限制都远远够不着。

## 5. 端点差异备注(不是本次改动)

官方文档示例用的 base URL 是 `https://api.moonshot.ai/v1`,而
`models.py:99 PROVIDER_BASE_URL["moonshot"]` 用的是 `https://api.moonshot.cn/v1`
(国内站)。两者是同一套 API 的不同接入点,视觉能力的文档挂在 .ai 站。
本次未改动 base URL —— 现网跑的是 .cn 且工作正常,换站是另一件事,
需要单独用 .cn 的模型列表复核后再动。

---

## 与代码的对应关系

| 事实 | 落在哪 |
|---|---|
| 哪些 id 支持视觉 | `agents/_shared/models.py: _VISION_MODELS` / `model_supports_vision()` |
| `image_url` + data URI 块格式 | `agents/_shared/skill_adapter.py` 出站时 materialize |
| data URI 从哪来 | `webui/scan_preview.py: render_scan_thumbnail()`(已带前缀,已按 path+mtime+size 缓存) |
| 不支持视觉时怎么办 | 降级成纯文本(不发 image 块),**不 400** |
