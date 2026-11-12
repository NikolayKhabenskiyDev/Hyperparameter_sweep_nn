from sklearn.model_selection import train_test_split
import json
import data.data_preparations as dp
import optim_experiments.series as op
import hyperparam_experiments.series as hyp

if __name__ == '__main__':
    scaled_df = dp.prepare_dataframe("data/Concrete_Data.xls", "data/data_diagrams")
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        scaled_df.iloc[:, :8], scaled_df.iloc[:, 8],
        test_size=0.4, random_state=42
    )
    X_valid, X_test, Y_valid, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.25, random_state=42
    )
    print(len(X_train), len(X_valid), len(X_test))

    results_op, best_parameters_op = op.optim_experiment_series(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
    print(results_op)

    results_hyp, best_parameters_hyp = hyp.optim_experiment_series(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
    print(results_hyp)

    with open("results/best_parameters_optimizer.json", "w", encoding="utf-8") as f:
        json.dump(best_parameters_op, f, indent=4, ensure_ascii=False)

    with open("results/best_parameters_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(best_parameters_hyp, f, indent=4, ensure_ascii=False)