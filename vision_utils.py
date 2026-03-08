import cv2
import numpy as np
import base64

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """画像のグレースケール化、平滑化、二値化など前処理を行う"""
    # BGR to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ぼかし処理（ノイズ除去）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 適応的二値化
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh

def detect_sticker_regions(processed_image: np.ndarray) -> list[tuple]:
    """シール候補領域のバウンディングボックスのリストを返す"""
    # 輪郭抽出
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    
    # 面積や形状でフィルタリング（シール領域の目安）
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 10000:  # 面積によるフィルタリング（サイズ調整が必要）
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        
        # シールはほぼ正方形に近い場合を想定
        if 0.5 <= aspect_ratio <= 2.0:
            bboxes.append((x, y, w, h))
            
    # X座標でソートする（左から右へ）
    bboxes.sort(key=lambda b: b[0])
    return bboxes

def np_image_to_base64(image: np.ndarray) -> str:
    """NumPy配列の画像をBase64文字列に変換する（Streamlitのデータエディタ表示用）"""
    _, buffer = cv2.imencode('.jpg', image)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def draw_bboxes(image: np.ndarray, results: list[dict]) -> np.ndarray:
    """結果（バウンディングボックスと点数）を元画像に描画する"""
    img_disp = image.copy()
    for res in results:
        x, y, w, h = res["bbox"]
        score = res.get("score")
        
        if res.get("is_valid"):
            color = (0, 255, 0) # Green for valid
            text = f"{score} pt"
        else:
            color = (0, 0, 255) # Red for invalid/error
            text = res.get("raw_text", "Err")
            
        # Draw bounding box
        cv2.rectangle(img_disp, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        cv2.putText(img_disp, text, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return img_disp
