"""
OCR调试工具 - 查看识别结果并测试修复方案
"""
import sys
import os
import mss
import time
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ocr_engine import OCREngine


def capture_test_image(region=None):
    """捕获一张测试图片"""
    if region is None:
        region = {"left": 100, "top": 100, "width": 800, "height": 600}
    
    print(f"正在捕获截图 (区域: {region})...")
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # 保存到文件
        img.save("debug_capture.png")
        print("已保存到: debug_capture.png")
        return img


def test_ocr():
    """测试OCR识别"""
    print("\n" + "="*60)
    print("OCR 调试工具")
    print("="*60)
    
    print("\n正在初始化OCR引擎...")
    engine = OCREngine(use_gpu=False, lang='ch', show_log=True)
    
    print("\n请准备好要识别的题目，3秒后开始捕获...")
    for i in range(3, 0, -1):
        print(i, end="...")
        time.sleep(1)
    print()
    
    img = capture_test_image()
    
    print("\n正在识别...")
    result = engine.recognize(img)
    
    print("\n" + "="*60)
    print("识别结果")
    print("="*60)
    print(f"is_valid: {result.get('is_valid')}")
    print(f"confidence: {result.get('confidence'):.3f}")
    print(f"error: {result.get('error')}")
    
    print("\n--- 原始文本 ---")
    print(result.get('raw_text', ''))
    
    print("\n--- 题目 ---")
    print(result.get('question', ''))
    
    print("\n--- 选项 ---")
    options = result.get('options', {})
    for k, v in options.items():
        print(f"{k}: {v}")
    
    if not options:
        print("\n⚠️  没有识别到任何选项！")
        print("\n尝试方案：降低有效选项数量要求...")
        
        # 测试修复方案
        from unittest.mock import patch
        import ocr_engine as ocr_module
        
        with patch.object(ocr_module.OCREngine, '_parse_question') as mock_parse:
            def patched_parse(self, text):
                if not text:
                    return self._empty_result("text is empty")
                text = '\n'.join(
                    re.sub(r'[^\S\n]+', ' ', line).strip()
                    for line in text.split('\n')
                )
                options_dict = {}
                for letter in ['A', 'B', 'C', 'D']:
                    option_content = self._extract_option(text, letter)
                    if option_content:
                        options_dict[letter] = option_content[:150].strip()
                question = self._extract_question(text, options_dict)
                return {
                    "question": question[:300],
                    "options": options_dict,
                    "raw_text": text,
                    "is_valid": len(options_dict) >= 1,  # 改为只要1个选项就有效
                    "confidence": 0.0
                }
            
            import re
            mock_parse.side_effect = lambda self, text: patched_parse(engine, text)
            
            print("\n--- 使用修复方案重新识别 ---")
            result2 = engine.recognize(img)
            
            print(f"\nis_valid: {result2.get('is_valid')}")
            print(f"options: {result2.get('options')}")
            print(f"question: {result2.get('question')}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_ocr()
