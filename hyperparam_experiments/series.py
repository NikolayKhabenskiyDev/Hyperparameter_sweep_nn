import hyperparam_experiments.functions as hyperparam_f

def optim_experiment_series(X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
    results = {}

    best_train, min_error, best_ep, best_bs = hyperparam_f.find_best_batch(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        [1, 20, 40, 60, 100, 200]
        )

    results['Batch size'] = [best_train, min_error, best_ep]
    print(f"Best bs = {best_bs}")

    best_train, min_error, best_ep, best_activation = hyperparam_f.find_best_activation(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        ["logistic", "tanh", "linear", "softsign", "softplus"]
        )

    results['Activation'] = [best_train, min_error, best_ep]
    print(f"Best activation = {best_activation}")

    best_train, min_error, best_ep, best_structure = hyperparam_f.find_best_structure(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        [[2, 2], [5, 5], [10, 10], [30, 10], [10, 30], [30, 30]]
        )

    results['Structure'] = [best_train, min_error, best_ep]

    best_train, min_error, best_ep, best_init = hyperparam_f.find_best_weight_init(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        ["xavier", "normal", "truncated_normal"]
        )

    results['Weight initialization'] = [best_train, min_error, best_ep]
    print(f"Best initialization = {best_init}")

    best_train, min_error, best_ep, best_reg = hyperparam_f.find_best_regularisation(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        [("l1", 0.01), ("l2", 0.01), ("l1", 0.1), ("l2", 0.1), ("l1", 0.5), ("l2", 0.5), ("l1", 1), ("l2", 1)]
        )

    results['Regularisation'] = [best_train, min_error, best_ep]
    print(f"Best regularisation = {best_reg}")

    best_parameters = (best_bs, best_activation, best_init, best_reg)
    return results, best_parameters
