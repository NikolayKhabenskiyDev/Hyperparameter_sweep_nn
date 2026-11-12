import numpy as np
import torch
import core.models as mod
import core.training as tr

OP_NAME = "GD"
def find_best_param(X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
                    candidates, make_net, label, param_name, training_func=tr.LearnWeightStatPlot, extra_parametrs={}):
    best_value = candidates[0]
    best_ep = 0
    min_err_test = float("inf")
    min_err_train = float("inf")

    for c in candidates:
        net = make_net(c)
        op = torch.optim.SGD(net.parameters(), lr=0.1)
        path = f"results/optim-{OP_NAME}/{param_name}-{label(c)}"

        if 'batch_size' in extra_parametrs:
            epochs, mean_err_train, mean_err_test = training_func(
                path, net, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, op, 300, b_size=c
                )
        elif 'regularization' in extra_parametrs:
            epochs, mean_err_train, mean_err_test = training_func(
                path, net, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, op, 300, reg_type=c[0], lambda_val=c[1]
                )
        else:
            epochs, mean_err_train, mean_err_test = training_func(
                path, net, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, op, 300
                )
        print("epoch: ", epochs, "error: ",
              round(mean_err_train, 3), round(mean_err_test, 3))

        if mean_err_test < min_err_test:
            min_err_test = mean_err_test
            min_err_train = mean_err_train
            best_value = c
            best_ep = epochs

    return min_err_train, min_err_test, best_ep, best_value

def find_best_batch(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, batch_sizes):
    return find_best_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test, batch_sizes,
        make_net=lambda _: mod.NeuralNetwork(),
        label=lambda b: b,
        param_name="batch size",
        extra_parametrs=["batch size"]
        )

def find_best_activation(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, functions):
    return find_best_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test, functions,
        make_net=lambda f: mod.NeuralNetwork(func=f),
        label=lambda f: f,
        param_name="activation"
        )

def find_best_structure(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, neuron_structures):
    return find_best_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test, neuron_structures,
        make_net=lambda s: mod.NeuralNetwork(neurons_linear1=s[0],
                                             neurons_linear2=s[1]),
        label=lambda s: str(s),
        param_name="neuron structure",
        training_func=tr.LearnSavePlot
        )

def find_best_weight_init(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, weight_inits):
    return find_best_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test, weight_inits,
        make_net=lambda i: mod.NeuralNetwork(weight_init_method=i),
        label=lambda i: i,
        param_name="weight initialization"
        )

def find_best_regularisation(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, weight_inits):
    return find_best_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test, weight_inits,
        make_net=lambda _: mod.NeuralNetwork(),
        label=lambda r: str(r),
        param_name="regularisation"
        )