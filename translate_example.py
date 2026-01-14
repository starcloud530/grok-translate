#!/usr/bin/env python3
"""
OpenAI完成式大模型翻译脚本示例（异步版本）
功能：
1. 定义翻译结果的BaseModel
2. JSON提取工具函数
3. Message构建函数
4. 异步翻译主函数
5. 异步示例用法
"""

from openai import AsyncOpenAI
from pydantic import BaseModel
import json
import asyncio
from typing import Dict, List, Optional


# ===============================
# 1. 定义翻译结果的BaseModel
# ===============================
class TranslateResult(BaseModel):
    """翻译结果的Pydantic模型"""
    translation_text: str


# ===============================
# 2. JSON提取工具函数
# ===============================
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
            print(f"❌ JSON解析失败，原始文本: {text[:100]}...")
            return None
    return None


# ===============================
# 3. Message构建函数
# ===============================
def build_translation_messages(source_text: str, target_lang: str, target_region: str) -> List[Dict[str, str]]:
    """
    构建翻译任务的message，与dataset.py中的格式保持一致
    
    Args:
        source_text: 源文本
        target_lang: 目标语言名称（如"Chinese"）
        target_region: 目标语言地区（如"China"）
    
    Returns:
        构建好的message列表
    """
    # 创建统一的system指令（与dataset.py保持一致）
    system_prompt = "You are a translation master, skilled at translating original text into the target language and returning your translated text."
    
    # 创建大模型指令（与dataset.py保持一致的格式）
    instruction = f"Translate original_text to {target_lang} language in {target_region} region.\n original_text:\n {source_text}\n Your Output is liked to be:\n {{\"translation_text\":\".....\"}}.**Use \"....\" to enclose your translated text to ensure your JSON output is complete. "
    
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


# ===============================
# 4. 异步翻译主函数
# ===============================
async def translate_text(
    client: AsyncOpenAI,
    model: str,
    source_text: str,
    target_lang: str,
    target_region: str,
    temperature: float = 0.0
) -> Optional[str]:
    """
    使用OpenAI异步客户端进行翻译
    
    Args:
        client: OpenAI异步客户端实例
        model: 模型名称
        source_text: 源文本
        target_lang: 目标语言名称（如"Chinese"）
        target_region: 目标语言地区（如"China"）
        temperature: 生成温度
    
    Returns:
        翻译结果文本，如果翻译失败返回None
    """
    try:
        # 构建message（传递target_lang和target_region参数）
        messages = build_translation_messages(source_text, target_lang, target_region)
        
        # 异步调用OpenAI完成式API
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "translate_result",
                    "schema": TranslateResult.model_json_schema()
                }
            },
            max_tokens=1024,
        )
        
        # 获取响应内容
        response_text = completion.choices[0].message.content
        
        # 提取JSON
        json_result = extract_json_from_response(response_text)
        if json_result is None:
            return None
        
        # 提取翻译结果
        return json_result.get("translation_text", None)
        
    except Exception as e:
        print(f"❌ 翻译失败，错误信息: {e}")
        return None


# ===============================
# 5. 异步示例用法
# ===============================
async def main():
    # 配置OpenAI异步客户端
    client = AsyncOpenAI(
        api_key="your_xai_api_key_here",  # 替换为你的API密钥
        base_url="https://api.x.ai/v1"  # 替换为你的API地址
    )
    
    # 翻译参数
    model = "grok-4-1-fast-non-reasoning"  # 替换为你的模型名称
    source_text = "Hello, world! How are you today?"
    lang_code = "hi"  # 目标语言代码

    # 读取映射表
    # 加载语言代码映射表
    with open("./lang_code_map.json","r", encoding="utf-8") as f:
        lang_code_map = json.load(f)
        
        # 获取目标语言信息
        target_lang = lang_code_map[lang_code]["lang"]
        target_region = lang_code_map[lang_code]["region"]
        
    # 异步执行翻译
    print(f"📝 源文本: {source_text}")
    translation = await translate_text(
        client=client,
        model=model,
        source_text=source_text,
        target_lang=target_lang,  # 传递目标语言名称
        target_region=target_region  # 传递目标语言地区
    )
    
    # 输出结果
    if translation:
        print(f"✅ 翻译结果 ({lang_code}): {translation}")
    else:
        print("❌ 翻译失败")


if __name__ == "__main__":
    asyncio.run(main())