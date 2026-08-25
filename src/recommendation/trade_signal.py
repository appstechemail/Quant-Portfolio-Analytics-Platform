def generate_trade_signal(reco):

    if not reco:
        return "NO TRADE"

    if reco["Signal"] == "BUY" and reco["Confidence"] > 0.6:
        return "STRONG BUY"

    elif reco["Signal"] == "BUY":
        return "BUY"

    elif reco["Signal"] == "SELL" and reco["Confidence"] > 0.6:
        return "STRONG SELL"

    elif reco["Signal"] == "SELL":
        return "SELL"

    return "HOLD"
