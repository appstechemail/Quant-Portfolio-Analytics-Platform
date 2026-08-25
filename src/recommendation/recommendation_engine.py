def generate_final_recommendation(result):

    if result is None or result.empty:
        return {}

    latest = result.iloc[-1]

    return {
        "Signal": latest["Signal"],
        "Confidence": round(latest["Confidence"], 3),
        "Probability": round(latest["Probability"], 3),
        "Entry": round(latest["Close"], 2),
        "Target": round(latest["Take_Profit"], 2),
        "Stop_Loss": round(latest["Stop_Loss"], 2),
        "Risk_Reward": round(
            (latest["Take_Profit"] - latest["Close"]) /
            (latest["Close"] - latest["Stop_Loss"] + 1e-9), 2
        ),
        "Holding_Days": int(
            (latest["Exit_Date"] - latest["Action_Date"]).days
        )
    }
