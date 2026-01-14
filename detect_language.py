import re
import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text as mp_text
from dotenv import load_dotenv

load_dotenv()

# ================== 模型下载与检查 ==================
def check_and_download_model():
    """检查模型是否存在，不存在则下载"""
    model_path = './models/language_detector.tflite'
    
    # 检查模型是否存在
    if os.path.exists(model_path):
        print(f"模型已存在: {model_path}")
        return True
    
    # 模型不存在，尝试下载
    print(f"模型不存在，正在下载...")
    
    # 使用下载脚本
    try:
        import urllib.request
        import shutil
        
        # 创建models目录
        os.makedirs('./models', exist_ok=True)
        
        # 从URL下载模型
        model_url = "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite"
        print(f"正在从 {model_url} 下载模型...")
        
        with urllib.request.urlopen(model_url) as response, open(model_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        
        if os.path.exists(model_path):
            print(f"模型下载成功! 文件大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            print("模型下载失败: 文件未创建")
            return False
    except Exception as e:
        print(f"模型下载失败: {e}")
        return False

# 检查并下载模型
if not check_and_download_model():
    print("无法获取模型文件，程序退出")
    sys.exit(1)

# ================== 初始化 MediaPipe ==================
MODEL_PATH = './models/language_detector.tflite'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_text.LanguageDetectorOptions(base_options=base_options)
detector = mp_text.LanguageDetector.create_from_options(options)

# ================== FastAPI ==================
app = FastAPI()
DEF_SRC_LNG = 'en'

class DetectionRequest(BaseModel):
    text: str

class DetectionNewRequest(BaseModel):
    text: str
    language_code: str




def detect_core(text: str) -> str:
    """核心检测逻辑：取最后 2 句检测语言（用 Google mediapipe）"""
    pattern = r'[^。！？!?.,，]+[。！？!?.,，]?'
    sentences = re.findall(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    selected_sentences = sentences[-2:]
    detect_sentence = " ".join(selected_sentences).replace("\n", " ").strip()

    result = detector.detect(detect_sentence)
    if not result.detections:
        return DEF_SRC_LNG
    # 取概率最大的语言
    best = max(result.detections, key=lambda d: d.probability)
    return best.language_code


@app.post("/detect")
def detect_api(req: DetectionRequest):
    """单检测"""
    try:
        text =req.text
        lang_code = detect_core(text)
        return {
            "code": 0,
            "msg": "success",
            "data": {"detected_language": lang_code}
        }
    except Exception as e:
        return {
            "code": -1,
            "msg": f"error: {e}",
            "data": {"detected_language": DEF_SRC_LNG}
        }

@app.post("/detect_new")
def detect_new_api(req: DetectionNewRequest):
    """简化版：直接检测 + is_need_llm 判断"""
    try:
        text = req.text
        result = detector.detect(text)

        if not result.detections:
            return {
                "code": 0,
                "msg": "success",
                "data": {
                    "detected_language": DEF_SRC_LNG,
                    "is_need_llm": True  # 默认需要
                }
            }
        # 取概率最大
        best = max(result.detections, key=lambda d: d.probability)

        # 判断是否需要 LLM
        is_need_llm = best.language_code != req.language_code

        return {
            "code": 0,
            "msg": "success",
            "data": {
                "detected_language": best.language_code,
                "is_need_llm": is_need_llm
            }
        }
    except Exception as e:
        return {
            "code": -1,
            "msg": f"error: {e}",
            "data": {
                "detected_language": DEF_SRC_LNG,
                "is_need_llm": True
            }
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DETECT_PORT", "9999"))
    uvicorn.run("detect_language:app", host="0.0.0.0", port=port, workers=2)