# Grok-4.1 翻译服务

一个基于 Grok-4.1 模型的高性能翻译服务，支持单条翻译和批量翻译功能.

## 功能特性

- ✅ **单条翻译与批量翻译**：支持单文本翻译和多文本批量翻译，批量翻译采用异步并发处理
- ✅ **语种自动检测**：智能判断源文本是否需要翻译，避免对已处于目标语言的文本进行翻译
- ✅ **多语言支持**：通过配置文件`lang_code_map.json`配置启用语言
- ✅ **高性能异步架构**：基于 FastAPI 和异步编程，提供卓越的并发处理能力
- ✅ **错误处理**：翻译失败时返回原文作为兜底，提供详细的状态码和错误消息
- ✅ **健康检查接口**：提供服务健康状态检查接口，便于监控和运维

## 技术栈

- **框架**：FastAPI
- **服务器**：Uvicorn
- **异步 HTTP 客户端**：aiohttp
- **API 客户端**：AsyncOpenAI
- **配置管理**：python-dotenv
- **数据验证**：Pydantic

## 安装与运行

### 1. 安装依赖

```bash
cd grok-translate-server
pip install -r requirements.txt
```

### 2. 配置环境变量

复制并编辑 `.env` 文件，设置必要的配置参数：

```bash
# 复制示例配置文件（如果不存在）
cp .env.example .env

# 编辑配置文件
vim .env
```

主要配置参数说明：

```dotenv
# API 配置
API_KEY="your-api-key"               # X.ai API 密钥
BASE_URL="https://api.x.ai/v1"        # X.ai API 基础 URL
MODEL="grok-4-1-fast-non-reasoning"   # 使用的模型名称
TEMPERATURE=0.0                       # 模型生成温度，0.0 表示更确定性的输出
MAX_TOKENS=1024                       # 最大生成 tokens 数
OPENAI_TIMEOUT=10                     # API 调用超时时间（秒）

# 语种检测配置
DETECT_URL="http://1.180.30.42:7008/detect_new"  # 语种检测服务 URL

# 服务器配置
HOST="0.0.0.0"                       # 服务器监听地址
PORT=8000                             # 服务器监听端口
LOG_LEVEL="info"                      # 日志级别

# 路径配置
LANGUAGE_MAP_PATH="./lang_code_map.json"  # 语言代码映射表路径

# 提示词配置
SYSTEM_PROMPT="You are a translation master..."  # 系统提示词
TRANSLATION_INSTRUCTION="Translate original_text to {target_lang}..."  # 翻译指令模板
```

### 3. 启动服务

```bash
python main.py
```

服务将在配置的端口上运行（默认 8000 端口）。

## API 接口文档

### 1. 单条翻译接口

#### 请求

```http
POST /v2/translate
Content-Type: application/json

{
    "text": "Hello, world!",
    "lang": "zh",
    "stream": false
}
```

#### 参数说明

- `text`：待翻译的文本内容
- `lang`：目标语言代码（如 "zh" 表示中文）
- `stream`：是否使用流式响应（目前不支持，固定为 false）

#### 响应

```json
{
    "code": 0,
    "msg": "",
    "choices": [
        {
            "message": {
                "content": "你好，世界！"
            }
        }
    ]
}
```

### 2. 批量翻译接口

#### 请求

```http
POST /v2/translate/batch
Content-Type: application/json

{
    "batch_inputs": {
        "1": {"input": "Hello"},
        "2": {"input": "World"}
    },
    "lang": "zh",
    "stream": false
}
```

#### 参数说明

- `batch_inputs`：批量翻译的输入文本，key 为任务标识，value 为包含 `input` 字段的对象
- `lang`：目标语言代码
- `stream`：是否使用流式响应（目前不支持，固定为 false）

#### 响应

```json
{
    "code": 0,
    "msg": "",
    "data": {
        "results": {
            "1": {
                "input": "Hello",
                "output": "你好",
                "code": 0,
                "msg": ""
            },
            "2": {
                "input": "World",
                "output": "世界",
                "code": 0,
                "msg": ""
            }
        }
    }
}
```

### 3. 健康检查接口

#### 请求

```http
GET /health
```

#### 响应

```json
{
    "status": "ok",
    "message": "Grok-4.1 Translation Service is running"
}
```

## 语言支持

支持的语言列表和配置定义在 `lang_code_map.json` 文件中。每种语言包含以下信息：

```json
"zh": {
    "lang_code": "zh",
    "lang": "Chinese",
    "region": "China",
    "enabled": true
}
```

- `lang_code`：语言代码
- `lang`：语言名称（英文）
- `region`：语言使用地区
- `enabled`：是否启用该语言

## 测试示例

### 单条翻译测试

```bash
curl -X POST "http://localhost:8000/v2/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, world!", "lang": "zh", "stream": false}'
```

### 批量翻译测试

```bash
curl -X POST "http://localhost:8000/v2/translate/batch" \
     -H "Content-Type: application/json" \
     -d '{"batch_inputs": {"1": {"input": "Hello"}, "2": {"input": "World"}}, "lang": "zh", "stream": false}'
```

### 健康检查测试

```bash
curl "http://localhost:8000/health"
```

## 注意事项

1. **API 密钥安全**：请妥善保管您的 X.ai API 密钥，避免泄露
2. **语种检测服务**：确保配置的语种检测服务 URL 可正常访问，否则将默认对所有文本进行翻译
3. **性能优化**：对于大规模批量翻译请求，建议合理控制并发数，避免超出 API 限制
4. **错误处理**：服务已实现自动重试和失败兜底机制，但仍建议在客户端实现适当的错误处理逻辑
5. **日志监控**：建议监控服务日志，及时发现和解决潜在问题

## 项目结构

```
grok-4.1-translate-server/
├── main.py              # 主程序入口
├── requirements.txt     # 项目依赖
├── .env                 # 环境变量配置
├── lang_code_map.json   # 语言代码映射表
├── translate_example.py # 翻译示例脚本
├── test/               # 测试目录
│   └── detect_lang_example.py  # 语种检测示例
└── README.md           # 项目说明文档
```

## 开发与扩展

### 添加新语言支持

编辑 `lang_code_map.json` 文件，添加新的语言配置：

```json
"your-lang-code": {
    "lang_code": "your-lang-code",
    "lang": "Language Name",
    "region": "Region Name",
    "enabled": true
}
```

### 自定义翻译提示词

编辑 `.env` 文件中的 `SYSTEM_PROMPT` 和 `TRANSLATION_INSTRUCTION` 变量，自定义翻译提示词：

```dotenv
SYSTEM_PROMPT="You are a professional translator..."
TRANSLATION_INSTRUCTION="Translate the following text to {target_lang}..."
```

## 故障排除

### 常见问题

1. **服务无法启动**
   - 检查端口是否被占用
   - 检查 API 密钥是否正确配置
   - 查看日志输出，定位具体错误

2. **翻译失败**
   - 检查网络连接是否正常
   - 检查 API 密钥是否有效
   - 检查目标语言是否已启用
   - 查看服务日志，了解具体错误原因

3. **语种检测不工作**
   - 检查语种检测服务 URL 是否可访问
   - 验证语种检测服务的 API 格式是否与预期一致

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请联系项目维护人员。