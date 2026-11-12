import torch
import core.models as mod
import core.training as tr
import optim_realisations.levenberg_marquardt as lm

def test_chosen_optims(X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
                       optimizer_factories, training_func):
    results = []
    for name, factory in optimizer_factories.items():
        net = mod.NeuralNetwork()
        optim = factory(net.parameters())
        epochs, mean_err_train, mean_err_test = training_func(
            f"results/optim-{name}", net,
            X_train, Y_train, X_valid, Y_valid, X_test, Y_test, optim, 300
            )
        print(f"Optimizer: {name}")
        print("epoch: ", epochs, "loss: ", round(mean_err_train, 3), round(mean_err_test, 3))

        results.append([mean_err_train, mean_err_test, epochs, optim])
    return results

def find_best_optim_param(X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
                          candidates, make_optimizer, make_path,
                          learn_fn=tr.LearnWeightStatPlot):
    best_value = candidates[0]
    best_ep = 0
    min_err_test = float("inf")
    min_err_train = float("inf")

    for c in candidates:
        net = mod.NeuralNetwork()
        optimizer = make_optimizer(net, c)
        path = make_path(c)

        epochs, mean_err_train, mean_err_test = learn_fn(
            path, net, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, optimizer, 300
            )
        print("epoch: ", epochs, "error: ",
              round(mean_err_train, 3), round(mean_err_test, 3))

        if mean_err_test < min_err_test:
            min_err_test = mean_err_test
            min_err_train = mean_err_train
            best_value = c
            best_ep = epochs
            best_optim = optimizer

    return min_err_train, min_err_test, best_ep, best_value, best_optim

def find_best_lr(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, learning_rates):
    return find_best_optim_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        learning_rates,
        make_optimizer=lambda net, lr: torch.optim.SGD(net.parameters(), lr=lr),
        make_path=lambda lr: f"results/optim-GD/lr-{lr}",
        learn_fn=tr.LearnSavePlot,
        )


def find_best_momentum(X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
                       lr, momentums, optim_type="GDM"):
    def make_optimizer(net, momentum):
        if optim_type == "GDM":
            return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        if optim_type == "NAG":
            return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                                   nesterov=True)
        raise ValueError(f"Unknown optim type {optim_type!r}")

    return find_best_optim_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        momentums, make_optimizer,
        make_path=lambda m: f"results/optim-{optim_type}/lr-{lr}, momentum-{m}",
        learn_fn=tr.LearnSavePlot,
        )


def find_best_smoothing(X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
                        lr, smoothing_consts, optim_type="RMSprop"):
    def make_optimizer(net, sm):
        if optim_type == "RMSprop":
            return torch.optim.RMSprop(net.parameters(), lr=lr, alpha=sm)
        if optim_type == "Adadelta":
            return torch.optim.Adadelta(net.parameters(), lr=lr, rho=sm)
        raise ValueError(f"Unknown optim type {optim_type!r}")

    return find_best_optim_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        smoothing_consts, make_optimizer,
        make_path=lambda sm: f"results/optim-{optim_type}/smoothing_const-{sm}",
        learn_fn=tr.LearnWeightStatPlot,
        )


def find_best_beta(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, lr, beta_coefs):
    return find_best_optim_param(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        beta_coefs,
        make_optimizer=lambda net, b: torch.optim.Adam(net.parameters(), lr=lr, betas=b),
        make_path=lambda b: f"results/optim-Adam/beta_coef-{b}",
        learn_fn=tr.LearnWeightStatPlot,
        )

def test_LM(X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
    net = mod.NeuralNetwork()
    epochs, mean_err_train, mean_err_test = lm.LearnLMPlot(
        "results/optim-LM", net, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, 300
        )
    return epochs, mean_err_train, mean_err_test