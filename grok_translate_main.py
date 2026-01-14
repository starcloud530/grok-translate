#!/usr/bin/env python3
"""
Grok-4.1翻译服务
功能：
1. 提供与seed-x模型完全兼容的翻译API接口
2. 支持单条翻译和批量翻译
3. 支持多语言翻译
4. 使用FastAPI构建高性能服务
5. 集成语种检测功能，避免不必要的翻译
"""

import json
import logging
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientResponseError, ServerTimeoutError, ServerDisconnectedError, ClientConnectionError, ClientConnectorError, ClientOSError, ClientPayloadError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="Grok-4.1 Translation Service", 
              description="提供与seed-x模型兼容的翻译API接口",
              version="1.0.0")

# ------------------------
# 数据模型定义
# ------------------------

class TranslationRequest(BaseModel):
    """单条翻译请求模型"""
    text: str = Field(..., description="待翻译的文本", alias="text")
    lang: str = Field(..., description="目标语言代码", alias="lang")
    stream: bool = Field(False, description="是否使用流式响应", alias="stream")
    
    model_config = {
        "populate_by_name": True,
    }

class BatchInputItem(BaseModel):
    """批量翻译输入项"""
    input: str = Field(..., description="待翻译的文本", alias="input")
    
    model_config = {
        "populate_by_name": True,
    }

class BatchModelRequest(BaseModel):
    """批量翻译请求模型"""
    batch_inputs: Dict[str, BatchInputItem] = Field(..., description="批量翻译输入", alias="batch_inputs")
    lang: str = Field(..., description="目标语言代码", alias="lang")
    stream: bool = Field(False, description="是否使用流式响应", alias="stream")
    
    model_config = {
        "populate_by_name": True,
    }

class BatchResultItem(BaseModel):
    """批量翻译结果项"""
    input: str = Field(..., description="原始文本", alias="Input")
    output: str = Field(..., description="翻译后的文本", alias="Output")
    code: int = Field(0, description="结果码，0表示成功", alias="Code")

class TranslateResult(BaseModel):
    """翻译结果的Pydantic模型"""
    translation_text: str

# 语种检测响应模型
class DetectLanguageResponse(BaseModel):
    code: int
    msg: str
    data: Dict[str, Any]

# ------------------------
# 配置加载
# ------------------------

# 加载语言代码映射表
def load_language_map() -> Dict[str, Dict[str, Union[str, bool]]]:
    """加载语言代码映射表，仅返回启用的语言"""
    try:
        language_map_path = os.getenv("LANGUAGE_MAP_PATH", "./lang_code_map.json")
        with open(language_map_path, "r", encoding="utf-8") as f:
            all_languages = json.load(f)
            # 只返回enabled字段为true的语言配置
            enabled_languages = {k: v for k, v in all_languages.items() if v.get("enabled", True)}
            return enabled_languages
    except Exception as e:
        logger.error(f"加载语言映射表失败: {e}")
        return {}

# 加载配置
def load_config() -> Dict[str, Any]:
    """加载服务配置"""
    detect_port = os.getenv("DETECT_PORT", "9999")
    config = {
        "api_key": os.getenv("API_KEY", ""),
        "base_url": os.getenv("BASE_URL", "https://api.x.ai/v1"),
        "model": os.getenv("MODEL", "grok-4-1-fast-non-reasoning"),
        "temperature": float(os.getenv("TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "1024")),
        "openai_timeout": int(os.getenv("OPENAI_TIMEOUT", 10)),
        "detect_port": detect_port,  # 语种检测服务端口
        "detect_url": os.getenv("DETECT_URL", "http://localhost:{DETECT_PORT}/detect_new").format(DETECT_PORT=detect_port),  # 语种检测服务URL
        "detect_timeout": int(os.getenv("DETECT_TIMEOUT", 5)),  # 语种检测超时时间
        "http_client_timeout": int(os.getenv("HTTP_CLIENT_TIMEOUT", 30)),  # HTTP客户端全局超时
        "http_connect_timeout": int(os.getenv("HTTP_CONNECT_TIMEOUT", 10)),  # HTTP连接超时
        "http_max_connections": int(os.getenv("HTTP_MAX_CONNECTIONS", 100)),  # 最大连接数
        "http_max_per_host": int(os.getenv("HTTP_MAX_PER_HOST", 30)),  # 每个主机的最大连接数
        "http_keepalive_timeout": int(os.getenv("HTTP_KEEPALIVE_TIMEOUT", 30))  # 保持连接超时
    }
    return config

# 全局配置
LANGUAGE_MAP = load_language_map()
CONFIG = load_config()

# OpenAI API配置
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL")
temperature = float(os.getenv("TEMPERATURE", 0.0))
max_tokens = int(os.getenv("MAX_TOKENS", 1024))
openai_timeout = int(os.getenv("OPENAI_TIMEOUT", 10))  # 从.env读取超时时间，默认10秒

# 初始化OpenAI客户端
client = AsyncOpenAI(
    api_key=CONFIG["api_key"],
    base_url=CONFIG["base_url"],
    timeout=CONFIG["openai_timeout"]  # 设置API超时时间
)

# 初始化aiohttp客户端会话
async def get_aiohttp_session():
    """获取或创建aiohttp客户端会话"""
    if not hasattr(get_aiohttp_session, "session"):
        # 配置超时参数
        timeout = ClientTimeout(
            total=CONFIG["http_client_timeout"],  # 总超时时间
            connect=CONFIG["http_connect_timeout"],  # 连接超时
            sock_connect=CONFIG["http_connect_timeout"],  # socket连接超时
            sock_read=CONFIG["http_client_timeout"]  # socket读取超时
        )
        
        # 创建连接器配置
        connector = aiohttp.TCPConnector(
            limit=CONFIG["http_max_connections"],  # 最大连接数
            limit_per_host=CONFIG["http_max_per_host"],  # 每个主机的最大连接数
            keepalive_timeout=CONFIG["http_keepalive_timeout"],  # 保持连接超时
            enable_cleanup_closed=True,  # 清理关闭的连接
            force_close=False,  # 不强制关闭连接
            ttl_dns_cache=300,  # DNS缓存时间（秒）
        )
        
        # 创建会话
        get_aiohttp_session.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            trust_env=False,  # 不使用系统代理设置
            raise_for_status=False  # 不自动抛出HTTP错误
        )
        
        logger.info(f"[HTTP客户端] 会话已创建: max_connections={CONFIG['http_max_connections']}, max_per_host={CONFIG['http_max_per_host']}, timeout={CONFIG['http_client_timeout']}s")
    
    return get_aiohttp_session.session

# ------------------------
# 工具函数
# ------------------------

async def detect_language(text: str, target_lang: str) -> Tuple[bool, Optional[str]]:
    """
    调用语种检测接口，判断是否需要翻译
    
    Args:
        text: 待检测的文本
        target_lang: 目标语言代码
        
    Returns:
        Tuple[bool, Optional[str]]: (is_need_llm, detected_language)
        如果is_need_llm为True，则需要翻译；否则不需要翻译
    """
    import time
    import traceback
    
    try:
        session = await get_aiohttp_session()
        
        # 记录请求开始时间
        start_time = time.time()
        
        logger.info(f"[语种检测] 开始请求: url={CONFIG['detect_url']}, text_len={len(text)}, target_lang={target_lang}")
        logger.info(f"[语种检测] 请求参数: text_preview={text[:100]}, language_code={target_lang}")
        logger.info(f"[语种检测] 超时配置: timeout={CONFIG['detect_timeout']}s")
        
        # 调用语种检测接口
        async with session.post(
            CONFIG["detect_url"],
            json={"text": text, "language_code": target_lang},
            timeout=CONFIG["detect_timeout"]
        ) as response:
            # 记录连接建立时间
            connect_time = time.time() - start_time
            logger.info(f"[语种检测] 连接建立成功: 耗时={connect_time:.3f}s")
            
            logger.info(f"[语种检测] 收到响应: status={response.status}")
            logger.info(f"[语种检测] 响应头: {dict(response.headers)}")
            
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"[语种检测] HTTP错误: status={response.status}, body={error_text[:500]}")
                return True, None  # 检测失败时默认需要翻译
            
            # 解析响应
            result = await response.json()
            
            if result["code"] != 0:
                logger.error(f"[语种检测] 业务错误: code={result['code']}, msg={result['msg']}")
                return True, None  # 检测失败时默认需要翻译
            
            # 返回检测结果
            detected_lang = result["data"].get("detected_language", None)
            is_need_llm = result["data"].get("is_need_llm", True)
            
            total_time = time.time() - start_time
            logger.info(f"[语种检测] 检测成功: detected={detected_lang}, need_llm={is_need_llm}, 总耗时={total_time:.3f}s")
            return is_need_llm, detected_lang
            
    except asyncio.TimeoutError:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] 请求超时: 超过{CONFIG['detect_timeout']}秒未收到响应, 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # 超时异常时默认需要翻译
    except ServerTimeoutError:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] 服务器超时: 检测服务响应超过{CONFIG['detect_timeout']}秒, 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # 超时异常时默认需要翻译
    except ClientConnectorError as e:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] TCP连接失败: 无法连接到服务器 {CONFIG['detect_url']}, 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] 错误类型={type(e).__name__}, 错误详情={e}")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # 连接失败时默认需要翻译
    except ServerDisconnectedError as e:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] 服务器断开连接: 检测服务在连接后断开, 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] 错误详情={e}")
        logger.error(f"[语种检测] 请求URL: {CONFIG['detect_url']}")
        logger.error(f"[语种检测] 请求Info: text_len={len(text)}, language_code={target_lang}")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # 连接断开时默认需要翻译
    except ClientOSError as e:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] 操作系统网络错误: 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] 错误类型={type(e).__name__}, 错误详情={e}")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # 操作系统网络错误时默认需要翻译
    except ClientPayloadError as e:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] 响应数据错误: 接收到无效的响应数据, 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] 错误详情={e}")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # 响应数据错误时默认需要翻译
    except ClientConnectionError as e:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] 连接异常: 连接过程中发生错误, 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] 错误类型={type(e).__name__}, 错误详情={e}")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # 连接异常时默认需要翻译
    except ClientResponseError as e:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] HTTP响应错误: 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] status={e.status}, message={e.message}, url={e.request_info.url}")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # HTTP响应错误时默认需要翻译
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌[语种检测] 未知异常: 已耗时={total_time:.3f}s")
        logger.error(f"[语种检测] 异常类型={type(e).__name__}, 错误信息={e}")
        logger.error(f"[语种检测] 堆栈信息:\n{traceback.format_exc()}")
        return True, None  # 其他异常时默认需要翻译

def extract_json_from_response(text: str) -> Optional[Dict]:
    """
    从模型响应文本中提取JSON内容
    
    Args:
        text: 模型返回的文本内容
    
    Returns:
        提取到的JSON字典，如果提取失败返回None
    """
    # 移除可能的代码块标记
    text = text.replace("```", "").replace("json", "")
    
    try:
        # 尝试直接解析整个文本
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # 尝试提取文本中的第一个JSON对象
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_content = text[json_start:json_end].replace("\\_", "_")
                return json.loads(json_content)
        except json.JSONDecodeError:
            # 解析失败
            logger.error(f"JSON解析失败，原始文本: {text[:100]}...")
            return None
    return None


def build_translation_messages(source_text: str, target_lang: str, target_region: str) -> List[Dict[str, str]]:
    """
    构建翻译任务的message
    
    Args:
        source_text: 源文本
        target_lang: 目标语言名称（如"Chinese"）
        target_region: 目标语言地区（如"China"）
    
    Returns:
        构建好的message列表
    """
    # 创建统一的system指令
    system_prompt = os.getenv("SYSTEM_PROMPT", "You are a translation master, skilled at translating original text into the target language and returning your translated text.")
    
    # 创建大模型指令
    instruction_template = os.getenv("TRANSLATION_INSTRUCTION", "Translate original_text to {target_lang} language in {target_region} region.\n original_text:\n {source_text}\n Your Output is liked to be:\n {{\"translation_text\":\".....\"}}.**Use \"....\" to enclose your translated text to ensure your JSON output is complete. ")
    instruction = instruction_template.format(target_lang=target_lang, target_region=target_region, source_text=source_text)
    
    # 构建完整的messages结构
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": instruction
        }
    ]


async def translate_text(source_text: str, lang_code: str) -> Optional[str]:
    """
    使用OpenAI异步客户端进行翻译
    
    Args:
        source_text: 源文本
        lang_code: 目标语言代码
    
    Returns:
        翻译结果文本，如果翻译失败返回None
    """
    try:
        logger.info(f"[翻译] 开始翻译: text_len={len(source_text)}, lang_code={lang_code}")
        
        # 1. 调用语种检测接口，判断是否需要翻译
        is_need_llm, detected_lang = await detect_language(source_text, lang_code)
        
        if not is_need_llm:
            # 如果不需要翻译（已经是目标语言），直接返回原文
            logger.info(f"[翻译] 跳过翻译: 已是目标语言")
            return source_text
        
        # 2. 获取目标语言信息
        if lang_code not in LANGUAGE_MAP:
            logger.error(f"[翻译] 不支持的语言代码: {lang_code}")
            return None
        
        lang_info = LANGUAGE_MAP[lang_code]
        target_lang = lang_info["lang"]
        target_region = lang_info["region"]
        
        # 3. 构建message
        messages = build_translation_messages(source_text, target_lang, target_region)
        
        logger.info(f"[翻译] 开始调用LLM: model={CONFIG['model']}, timeout={CONFIG['openai_timeout']}s")
        
        # 4. 异步调用OpenAI完成式API
        completion = await client.chat.completions.create(
            model=CONFIG["model"],
            messages=messages,
            temperature=CONFIG["temperature"],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "translate_result",
                    "schema": TranslateResult.model_json_schema()
                }
            },
            max_tokens=CONFIG["max_tokens"],
        )
        
        logger.info(f"[翻译] LLM响应成功")
        
        # 5. 获取响应内容
        response_text = completion.choices[0].message.content
        
        # 6. 提取JSON
        json_result = extract_json_from_response(response_text)
        if json_result is None:
            logger.error(f"[翻译] JSON解析失败: response_text={response_text[:100]}")
            return None
        
        # 7. 提取翻译结果
        translated_text = json_result.get("translation_text", None)
        logger.info(f"[翻译] 翻译成功: result_len={len(translated_text) if translated_text else 0}")
        return translated_text
        
    except asyncio.TimeoutError:
        logger.error(f"[翻译] 请求超时: LLM服务响应超过{CONFIG['openai_timeout']}秒")
        return None
    except ClientConnectorError as e:
        logger.error(f"[翻译] TCP连接失败: 无法连接到LLM服务 {CONFIG['base_url']}, 错误类型={type(e).__name__}, 错误详情={e}")
        return None
    except ServerDisconnectedError as e:
        logger.error(f"[翻译] 服务器断开连接: LLM服务在响应过程中断开连接, 错误详情={e}")
        return None
    except ClientOSError as e:
        logger.error(f"[翻译] 操作系统网络错误: 错误类型={type(e).__name__}, 错误详情={e}")
        return None
    except ClientPayloadError as e:
        logger.error(f"[翻译] 响应数据错误: 接收到无效的响应数据, 错误详情={e}")
        return None
    except ClientConnectionError as e:
        logger.error(f"[翻译] 连接异常: 连接LLM服务过程中发生错误, 错误类型={type(e).__name__}, 错误详情={e}")
        return None
    except ClientResponseError as e:
        logger.error(f"[翻译] HTTP响应错误: status={e.status}, message={e.message}, url={e.request_info.url}")
        return None
    except Exception as e:
        logger.error(f"[翻译] 未知异常: type={type(e).__name__}, error={e}")
        return None

# ------------------------
# API路由
# ------------------------

@app.post("/v2/translate", tags=["翻译"])
async def translate(request: TranslationRequest) -> JSONResponse:
    """
    单条翻译接口
    与seed-x模型API接口保持完全兼容
    优化：翻译失败时返回原文作为兜底，并添加状态码和错误消息
    """
    logger.info(f"收到单条翻译请求: Text={request.text[:50]}..., Lang={request.lang}, Stream={request.stream}")
    
    # 执行翻译
    translated_text = await translate_text(request.text, request.lang)
    
    # 构造响应
    if translated_text is not None:
        # 翻译成功
        code = 0
        msg = ""
        response_content = translated_text
        logger.info(f"单条翻译完成: Result={translated_text[:50]}...")
    else:
        # 翻译失败，返回原文作为兜底
        code = -1
        msg = "翻译失败"
        response_content = request.text
        logger.error(f"单条翻译失败，返回原文作为兜底: Text={request.text[:10]}...")
    
    response = {
        "code": code,
        "msg": msg,
        "choices": [
            {
                "message": {
                    "content": response_content
                }
            }
        ]
    }
    
    return JSONResponse(content=response)


@app.post("/v2/translate/batch", tags=["翻译"])
async def batch_translate(request: BatchModelRequest) -> JSONResponse:
    """
    批量翻译接口
    与seed-x模型API接口保持完全兼容
    优化：使用异步并发执行批量翻译任务
    """
    logger.info(f"收到批量翻译请求: 任务数={len(request.batch_inputs)}, Lang={request.lang}, Stream={request.stream}")
    
    if not request.batch_inputs:
        # 构造响应
        response = {
            "code": 0,
            "msg": "success",
            "data": {
                "results": {}
            }
        }
        return JSONResponse(content=response)
    
    # 创建异步任务列表
    async def translate_task(key: str, input_text: str) -> Tuple[str, Dict[str, Any]]:
        """单个翻译任务的包装函数"""
        translated_text = await translate_text(input_text, request.lang)
        
        if translated_text is None:
            return key, {
                "input": input_text,
                "output": input_text,  # 翻译失败时返回原文
                "code": -1,  # 翻译失败 code = -1
                "msg": "翻译失败"  # 添加错误消息
            }
        else:
            return key, {
                "input": input_text,
                "output": translated_text,
                "code": 0,  # 翻译成功 code = 0
                "msg": ""  # 成功时消息为空
            }
    
    # 并发执行所有翻译任务
    tasks = [translate_task(key, item.input) for key, item in request.batch_inputs.items()]
    results = {}
    
    try:
        # 使用asyncio.gather并发执行任务
        completed_tasks = await asyncio.gather(*tasks)
        
        # 将结果转换为字典
        for key, result in completed_tasks:
            results[key] = result
    except Exception as e:
        logger.error(f"批量翻译任务执行失败: {e}")
        # 如果整体任务失败，逐个处理
        for key, item in request.batch_inputs.items():
            results[key] = {
                "input": item.input,
                "output": item.input,
                "code": -1
            }
    
    # 构造响应
    # 计算整体状态：如果有任何任务失败，整体返回失败
    overall_success = all(result["code"] == 0 for result in results.values())
    overall_code = 0 if overall_success else -1
    overall_msg = "" if overall_success else "部分任务翻译失败"
    
    response = {
        "code": overall_code,
        "msg": overall_msg,
        "data": {
            "results": results
        }
    }
    
    logger.info(f"批量翻译完成: 任务数={len(request.batch_inputs)}, 成功={overall_success}")                        
    return JSONResponse(content=response)


@app.get("/health", tags=["健康检查"])
async def health_check() -> JSONResponse:
    """
    健康检查接口
    """
    return JSONResponse(content={"status": "ok", "message": "Grok-4.1 Translation Service is running"})


# 应用关闭时清理资源
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    if hasattr(get_aiohttp_session, "session"):
        await get_aiohttp_session.session.close()
    logger.info("应用关闭，资源已清理")


if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动Grok-4.1翻译服务...")
    logger.info(f"支持的语言数: {len(LANGUAGE_MAP)}")
    logger.info(f"语种检测接口: {CONFIG['detect_url']}")
    
    # 启动服务
    uvicorn.run(
        app, 
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", "8000")), 
        log_level=os.getenv("LOG_LEVEL", "info")
    )