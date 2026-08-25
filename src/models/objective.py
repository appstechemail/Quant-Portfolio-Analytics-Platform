def objective(params):

    model = build_model(params)

    wf_results = walkforward(
        model=model
    )

    score = (

        wf_results["Avg_Sharpe"]

        - 0.50 * abs(wf_results["Worst_DD"])

        - 0.05 * wf_results["Avg_Turnover"]

    )

    return score