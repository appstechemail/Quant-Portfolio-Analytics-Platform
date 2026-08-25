SEARCH_SPACE = {

    "xgb": {

        "n_estimators": (200, 1000),
        "max_depth": (3, 8),
        "learning_rate": (0.01, 0.10),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_weight": (1, 20),
        "gamma": (0.0, 5.0)

    },

    "rf": {

        "n_estimators": (200,1000),
        "max_depth": (3,10),
        "min_samples_leaf": (5,50)

    },

    "lr": {

        "C": (0.001,10)

    }

}