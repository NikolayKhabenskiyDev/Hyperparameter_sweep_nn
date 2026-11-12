import numpy as np
import torch
import os
import core.result_output as output
from torch.utils.data import DataLoader, TensorDataset

def GetPredAndError(network, X_tensor, Y_actual):
    Y_pred = network(X_tensor)
    Y_pred_array = Y_pred.detach().numpy().flatten()
    abs_errors = np.abs(np.array(Y_actual) - Y_pred_array)
    mean_error = np.mean(abs_errors)
    std_error = np.std(abs_errors)
    return Y_pred, mean_error, std_error

def Learn(
        network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
        optimizer, epochs, b_size=None, reg_type=None, lambda_val=0,
        input_noise=0, gradient_noise=0):
    loss_fn = torch.nn.MSELoss()
    loss_train, loss_valid, loss_test, err_train, err_valid, err_test = [], [], [], [], [], []
    min_loss = 100

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
    X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    Y_valid_tensor = torch.tensor(Y_valid.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

    if b_size == None:
        b_size=len(X_train_tensor)

    dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    loader = DataLoader(dataset, batch_size=b_size, shuffle=False)

    for ep in range(epochs):
        for x_batch, y_batch in loader:
            if input_noise > 0:
                noise = torch.normal(0, input_noise, size=x_batch.shape, device=x_batch.device)
                x_batch = x_batch + noise
            def closure():
                y_pred = network(x_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()

                if reg_type == 'l1':
                    l1_norm = sum(p.abs().sum() for p in network.parameters())
                    loss += lambda_val * l1_norm
                elif reg_type == 'l2':
                    l2_norm = (sum(p.pow(2).sum() for p in network.parameters()))
                    loss += lambda_val * l2_norm

                loss.backward()

                if gradient_noise > 0:
                    for param in network.parameters():
                        if param.grad is not None:
                            grad_noise = torch.normal(0, gradient_noise, size=param.grad.shape, device=param.grad.device)
                            param.grad.add_(grad_noise)
                return loss

            optimizer.step(closure)

        Y_pred, mean_error, _ = GetPredAndError(network, X_train_tensor, Y_train)
        loss_train.append(loss_fn(Y_pred, Y_train_tensor).item())

        Y_pred, mean_error, _ = GetPredAndError(network, X_valid_tensor, Y_valid)
        loss_valid.append(loss_fn(Y_pred, Y_valid_tensor).item())

        Y_pred, mean_error, _ = GetPredAndError(network, X_test_tensor, Y_test)
        loss_test.append(loss_fn(Y_pred, Y_test_tensor).item())

        if ep % 50 == 0:
            if loss_valid[-1] >= min_loss:
                break
            else:
                min_loss = loss_valid[-1]

    _, mean_err_train, _ = GetPredAndError(network, X_train_tensor, Y_train)

    _, mean_err_test, _ = GetPredAndError(network, X_test_tensor, Y_test)

    loss_dict = {"train loss": loss_train, "valid loss": loss_valid, "test loss": loss_test}

    return ep + 1, mean_err_train, mean_err_test, loss_dict

def LearnSaveText(
        save_path, network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
        optimizer, epochs, b_size=None, reg_type=None, lambda_val=0,
        input_noise=0, gradient_noise=0
    ):
    ep, mean_err_train, mean_err_test, loss_dict = Learn(
        network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
        optimizer, epochs, b_size, reg_type, lambda_val, input_noise, gradient_noise
        )

    os.makedirs(save_path, exist_ok=True)
    output.save_error_value(save_path, ep, mean_err_train, mean_err_test)
    return ep, mean_err_train, mean_err_test

def LearnSavePlot(
        save_path, network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
        optimizer, epochs, b_size=None, reg_type=None, lambda_val=0,
        input_noise=0, gradient_noise=0
    ):
    ep, mean_err_train, mean_err_test, loss_dict = Learn(
        network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
        optimizer, epochs, b_size, reg_type, lambda_val, input_noise, gradient_noise
        )

    os.makedirs(save_path, exist_ok=True)
    output.save_error_value(save_path, ep, mean_err_train, mean_err_test)
    output.save_plt(save_path, loss_dict, ep)
    return ep, mean_err_train, mean_err_test

def LearnSaveAggregation(
        save_path, network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
        optimizer, epochs, b_size=None, reg_type=None, lambda_val=0,
        input_noise=0, gradient_noise=0, repeat_amount=10):

    ep_list = []
    err_train_list = []
    err_test_list = []
    for ii in range(repeat_amount):
        ep, err_train, err_test, _ = Learn(
            network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
            optimizer, epochs, b_size, reg_type, lambda_val, input_noise, gradient_noise
            )
        ep_list.append(ep)
        err_train_list.append(err_train)
        err_test_list.append(err_test)

    mean_ep = np.mean(ep_list)
    mean_err_train = np.mean(err_train_list)
    mean_err_test = np.mean(err_test_list)
    output.save_error_value(save_path, mean_ep, mean_err_train, mean_err_test)
    return ep, mean_err_train, mean_err_test

W_INDEXES = [0, 181, 601]

def LearnWeightStat(network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, optimizer, epochs, w_indexes=None):
    if w_indexes == None:
        w_indexes = W_INDEXES
    loss_fn = torch.nn.MSELoss()
    loss_train, loss_valid, loss_test, err_train, err_valid, err_test = [], [], [], [], [], []
    all_weights = []
    delta1, delta2, delta3, mean_delta, max_delta, min_delta = [], [], [], [], [], []
    min_loss = 1000

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
    X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    Y_valid_tensor = torch.tensor(Y_valid.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

    for ep in range(epochs):
        def closure():
            y_pred = network(X_train_tensor)
            loss = loss_fn(y_pred, Y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)

        all_params = optimizer.param_groups[0]['params']
        all_weights.append(torch.cat([p.data.view(-1) for p in all_params]))

        if ep != 0:
            all_deltas = abs(all_weights[-2] - all_weights[-1])
            delta1.append(all_deltas[w_indexes[0]])
            delta2.append(all_deltas[w_indexes[1]])
            delta3.append(all_deltas[w_indexes[2]])
            mean_delta.append(all_deltas.mean())
            max_delta.append(all_deltas.max())
            min_delta.append(all_deltas.min())

        Y_pred, mean_error, _ = GetPredAndError(network, X_train_tensor, Y_train)
        loss_train.append(loss_fn(Y_pred, Y_train_tensor).item())

        Y_pred, mean_error, _ = GetPredAndError(network, X_valid_tensor, Y_valid)
        loss_valid.append(loss_fn(Y_pred, Y_valid_tensor).item())

        Y_pred, mean_error, _ = GetPredAndError(network, X_test_tensor, Y_test)
        loss_test.append(loss_fn(Y_pred, Y_test_tensor).item())

        if ep % 50 == 0:
            if loss_valid[-1] >= min_loss:
                break
            else:
                min_loss = loss_valid[-1]

    _, mean_err_train, _ = GetPredAndError(network, X_train_tensor, Y_train)
    _, mean_err_test, _ = GetPredAndError(network, X_test_tensor, Y_test)

    loss_dict = {"train loss": loss_train, "valid loss": loss_valid, "test loss": loss_test}
    delta_dict = {r"$w_{1}$": delta1, r"$w_{2}$": delta2, r"$w_{3}$": delta3, r"$overline{w}$": mean_delta, r"$w_{min}$": min_delta, r"$w_{max}$": max_delta}

    return ep + 1, mean_err_train, mean_err_test, loss_dict, delta_dict

def LearnWeightStatPlot(save_path, network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, optimizer, epochs, w_indexes=None):
    ep, mean_err_train, mean_err_test, loss_dict, delta_dict = LearnWeightStat(network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, optimizer, epochs, w_indexes)

    os.makedirs(save_path, exist_ok=True)
    output.save_error_value(save_path, ep, mean_err_train, mean_err_test)
    output.save_plt(save_path, loss_dict, ep)

    return ep, mean_err_train, mean_err_test