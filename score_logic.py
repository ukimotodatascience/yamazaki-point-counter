def calculate_total(results: list[dict]) -> float:
    """有効なシールデータの点数を合算して合計スコアを算出する"""
    total = 0.0
    for res in results:
        if res.get("is_valid") and res.get("score") is not None:
            try:
                total += float(res["score"])
            except ValueError:
                pass
    return total
