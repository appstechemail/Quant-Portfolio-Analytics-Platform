from src.models import objective

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=100
)