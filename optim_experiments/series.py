import torch
import optim_realisations.steepest_gd as sgd
import optim_realisations.ngd as ngd
import optim_realisations.levenberg_marquardt as lm
import optim_experiments.functions as optim_f
import core.models as mod
import core.training as tr

def optim_experiment_series(X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
    results = {}

    err_train, err_test, best_ep, best_lr, best_optim = optim_f.find_best_lr(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        [0.1, 0.05, 0.01, 0.005, 0.001]
        )

    results['GD'] = [err_train, err_test, best_ep, best_optim]

    print(f"Best lr = {best_lr}")

    err_train, err_test, best_ep, best_momentum_GDM, best_optim = optim_f.find_best_momentum(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        best_lr, [0.2, 0.4, 0.6, 0.8], "GDM"
        )

    results['GDM'] = [err_train, err_test, best_ep, best_optim]

    err_train, err_test, best_ep, best_momentum_NAG, best_optim = optim_f.find_best_momentum(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        best_lr, [0.2, 0.4, 0.6, 0.8], "NAG"
        )

    results['NAG'] = [err_train, err_test, best_ep, best_optim]

    print(f"Best momentum (GDM) = {best_momentum_GDM},"
          f"Best momentum (NAG) = {best_momentum_NAG}")

    optimizer_f = {
        "Adagrad": lambda p: torch.optim.Adagrad(p, lr=best_lr),
        "Rprop": lambda p: torch.optim.Rprop(p, lr=best_lr),
        "SGD": lambda p: sgd.SteepestGD(p),
        "FR": lambda p: ngd.NGD(p, method='FR', line_search='Armijo',
                                c1=1e-4, c2=0.5, lr=0.2, max_ls=25),
        "PRP": lambda p: ngd.NGD(p, method='PRP', line_search='Armijo',
                                 c1=1e-4, c2=0.9, lr=1, rho=0.5),
        "LBFGS": lambda p: torch.optim.LBFGS(p, lr=best_lr),
    }
    temp_results = optim_f.test_chosen_optims(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        optimizer_f, tr.LearnSavePlot
        )
    results['Adagrad'] = temp_results[0]
    results['Rprop'] = temp_results[1]
    results['Steepest GD'] = temp_results[2]
    results['FR'] = temp_results[3]
    results['PRP'] = temp_results[4]
    results['LBFGS'] = temp_results[5]

    err_train, err_test, best_ep, best_sm_const_RMSprop, best_optim = optim_f.find_best_smoothing(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        best_lr, [0.5, 0.7, 0.9, 0.99], 'RMSprop'
        )
    results['RMSprop'] = [err_train, err_test, best_ep, best_optim]

    err_train, err_test, best_ep, best_sm_const_Adadelta, best_optim = optim_f.find_best_smoothing(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
        best_lr, [0.5, 0.7, 0.9, 0.99], 'Adadelta'
        )
    results['Adadelta'] = [err_train, err_test, best_ep, best_optim]

    print(f"Best smoothing const (RMSprop) = {best_sm_const_RMSprop},"
          f"Best momentum (Adadelta) = {best_sm_const_Adadelta}")

    err_train, err_test, best_ep, best_beta_coef, best_optim = optim_f.find_best_beta(
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test, best_lr,
        [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.95), (0.95, 0.99)])
    results['Adam'] = [err_train, err_test, best_ep, best_optim]

    print(f"Best beta coef (Adam) = {best_beta_coef}")

    err_train, err_test, best_ep = lm.LearnLMPlot(
        f"results/optim-LM", mod.NeuralNetwork(),
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test, 300, best_lr
        )

    best_optim_name = min(results, key=lambda name: results[name][1])

    best_parameters = (
        best_lr, best_momentum_GDM, best_momentum_NAG,
        best_sm_const_RMSprop, best_sm_const_Adadelta,
        best_beta_coef, best_optim_name
                       )
    return results, best_parameters
