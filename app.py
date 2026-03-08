import streamlit as st
import cv2
import numpy as np
from PIL import Image

from vision_utils import preprocess_image, detect_sticker_regions, draw_bboxes, np_image_to_base64
from ocr_utils import run_ocr, normalize_score
from score_logic import calculate_total

# ページ設定
st.set_page_config(
    page_title="ヤマザキ春のパン祭り シール集計アプリ",
    page_icon="🍞",
    layout="centered"
)

# --- セッションステートの初期化 ---
if "raw_image" not in st.session_state:
    st.session_state["raw_image"] = None
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = []
if "total_score" not in st.session_state:
    st.session_state["total_score"] = 0.0

def process_uploaded_image(image_bytes):
    """アップロードされた画像の解析パイプラインを実行する"""
    # bytesからOpenCV画像に変換
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    st.session_state["raw_image"] = image
    
    with st.spinner('画像を解析中です...'):
        # 1. 前処理と領域検出
        processed_img = preprocess_image(image)
        bboxes = detect_sticker_regions(processed_img)
        
        results = []
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            
            # シール画像を切り出し
            cropped = image[y:y+h, x:x+w]
            cropped_base64 = np_image_to_base64(cropped)
            
            # 2. OCRと正規化
            raw_text = run_ocr(cropped)
            score = normalize_score(raw_text)
            
            is_valid = (score is not None)
            
            results.append({
                "id": i + 1,
                "bbox": bbox,
                "cropped_image_base64": cropped_base64,
                "raw_text": raw_text,
                "score": score if is_valid else raw_text, # エラー時は入力値そのまま表示
                "is_valid": is_valid
            })
            
        st.session_state["processed_results"] = results
        st.session_state["total_score"] = calculate_total(results)

def render_upload_section():
    """画像入力画面の描画"""
    st.title("🍞 春のパン祭り 点数自動集計")
    st.markdown("台紙全体が明るく、影が入らないように真上から撮影してください。")
    
    # カメラ連携
    camera_img = st.camera_input("カメラで撮影")
    # アップローダー
    uploaded_file = st.file_uploader("写真を選択", type=["jpg", "jpeg", "png"])
    
    if camera_img is not None:
        process_uploaded_image(camera_img.getvalue())
        st.rerun()
        
    elif uploaded_file is not None:
        process_uploaded_image(uploaded_file.getvalue())
        st.rerun()

def render_results_section():
    """解析結果の表示と手動修正画面の描画"""
    st.title("📊 解析結果")
    
    # リセット機能
    if st.button("もう一度撮影する"):
        st.session_state["raw_image"] = None
        st.session_state["processed_results"] = []
        st.session_state["total_score"] = 0.0
        st.rerun()
        
    results = st.session_state["processed_results"]
    raw_image = st.session_state["raw_image"]
    
    if len(results) == 0:
        st.warning("システムが点数シールを検出できませんでした。画像を再撮影するか、明るい場所で撮影し直してください。")
        st.image(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), caption="アップロードされた画像", use_container_width=True)
        return

    # メトリクスの表示
    total = st.session_state["total_score"]
    valid_count = sum(1 for r in results if r["is_valid"])
    error_count = len(results) - valid_count
    
    col1, col2, col3 = st.columns(3)
    col1.metric("合計点数", f"{total:.1f} 点")
    col2.metric("認識シール数", f"{valid_count} 枚")
    col3.metric("認識エラー", f"{error_count} 件")
    
    # プレビュー画像の生成
    disp_img = draw_bboxes(raw_image, results)
    st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), caption="検出結果プレビュー", use_container_width=True)
    
    st.subheader("結果の手動修正")
    st.markdown("点数が間違っている場合、表の **「score」** 列の数値を直接書き換えてください。合計に即時反映されます。")
    
    # data_editor用にリストを変換
    editor_data = []
    for r in results:
        editor_data.append({
            "id": r["id"],
            "サムネイル": r["cropped_image_base64"],
            "raw_text": r["raw_text"],
            "score": float(r["score"]) if r["is_valid"] and r["score"] != "" else r["score"],
            "is_valid": r["is_valid"]
        })
        
    # サムネイル（画像列）の表示設定
    edited_df = st.data_editor(
        editor_data,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "サムネイル": st.column_config.ImageColumn("画像", help="切り出されたシール画像"),
            "raw_text": st.column_config.TextColumn("OCR生テキスト", disabled=True),
            "score": st.column_config.NumberColumn("点数 (変更可)", required=True),
            "is_valid": st.column_config.CheckboxColumn("有効?", disabled=True)
        },
        hide_index=True,
        use_container_width=True,
    )
    
    # 編集結果を反映して再計算
    new_results = []
    has_changed = False
    for i, edited_row in enumerate(edited_df):
        orig_row = results[i]
        
        # 点数が手動で変更された場合
        if str(edited_row["score"]) != str(orig_row["score"]):
            has_changed = True
            try:
                # 浮動小数点数値に変換可能なら有効とする
                val = float(edited_row["score"])
                orig_row["score"] = val
                orig_row["is_valid"] = True
            except ValueError:
                # 変換不能な文字列になった場合は無効とする
                orig_row["score"] = edited_row["score"]
                orig_row["is_valid"] = False
                
        new_results.append(orig_row)
        
    if has_changed:
        st.session_state["processed_results"] = new_results
        st.session_state["total_score"] = calculate_total(new_results)
        st.rerun()

def main():
    if st.session_state["raw_image"] is None:
        render_upload_section()
    else:
        render_results_section()

if __name__ == "__main__":
    main()
