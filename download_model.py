#!/usr/bin/env python3
"""
下载语言检测模型脚本
"""
import os
import urllib.request
import shutil

def download_model():
    """下载语言检测模型"""
    # 创建models目录
    os.makedirs("models", exist_ok=True)
    
    # 模型下载URL和保存路径
    model_url = "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite"
    model_path = "./models/language_detector.tflite"
    
    # 检查模型是否已存在
    if os.path.exists(model_path):
        print(f"模型已存在: {model_path}")
        return True
    
    print(f"正在下载模型: {model_url}")
    print(f"保存路径: {model_path}")
    
    try:
        # 下载模型
        with urllib.request.urlopen(model_url) as response, open(model_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        
        print(f"模型下载成功! 文件大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"模型下载失败: {e}")
        return False

if __name__ == "__main__":
    download_model()
