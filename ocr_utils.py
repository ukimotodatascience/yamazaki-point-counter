import pytesseract
import numpy as np
import re

def run_ocr(cropped_img: np.ndarray) -> str:
    """切り出された画像区画からテキストを読み取る"""
    # OCRエンジンの設定
    # ※ 本番環境では pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' などのパス設定が必要な場合あり
    # Tesseractのページセグメンテーションモードを "単一文字" (10) や "単一単語" (8) に設定する
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0125.'
    
    try:
        text = pytesseract.image_to_string(cropped_img, config=custom_config)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def normalize_score(raw_text: str) -> float | None:
    """OCRの生テキストを点数（0.5, 1.0など）に正規化する"""
    if not raw_text:
        return None
        
    # 余分な空白や改行を削除
    text = re.sub(r'\s+', '', raw_text)
    
    # 代表的な誤認識パターンの補正
    text = text.replace("o", "0").replace("O", "0")
    
    # 完全に一致するパターンを処理
    if text in ["05", ".5", "o.5", "0.5", "0,5"]:
        return 0.5
    elif text in ["1", "1.0", "1.", "1,0"]:
        return 1.0
    elif text in ["15", "1.5", "1,5"]:
        return 1.5
    elif text in ["2", "2.0", "2.", "2,0"]:
        return 2.0
        
    # 正規表現でのマッチングも念のため
    try:
        score = float(text)
        if score in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]: # ありえるパン祭りの点数
            return score
    except ValueError:
        pass
        
    return None
