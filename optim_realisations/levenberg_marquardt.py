import torch
from torch.utils.data import DataLoader, TensorDataset
import torch_levenberg_marquardt as tlm
import core.training as tr
import core.result_output as output

def LearnLM(network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs, lr=1.0):

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
    X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    Y_valid_tensor = torch.tensor(Y_valid.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

    lm_module = tlm.training.LevenbergMarquardtModule(
        model=network,
        loss_fn=tlm.loss.MSELoss(),
        learning_rate=lr,
        attempts_per_step=10,
        solve_method='qr',
    )

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=len(X_train_tensor), shuffle=False)

    loss_train, loss_valid, loss_test = [], [], []
    all_weights = []

    min_loss = float("inf")

    for ep in range(epochs):
        tlm.utils.fit(lm_module, train_loader, epochs=1)

        curr_weights = torch.cat([p.data.view(-1) for p in network.parameters()]).clone()
        all_weights.append(curr_weights)

        with torch.no_grad():
            network.eval()
            loss_train.append(tlm.loss.MSELoss()(network(X_train_tensor), Y_train_tensor).item())
            loss_valid.append(tlm.loss.MSELoss()(network(X_valid_tensor), Y_valid_tensor).item())
            loss_test.append(tlm.loss.MSELoss()(network(X_test_tensor), Y_test_tensor).item())
            network.train()

        if ep % 50 == 0:
            if loss_valid[-1] >= min_loss:
                break
            else:
                min_loss = loss_valid[-1]

    _, mean_err_train, _ = tr.GetPredAndError(network, X_train_tensor, Y_train)
    _, mean_err_test, _ = tr.GetPredAndError(network, X_test_tensor, Y_test)

    loss_dict = {"train loss": loss_train, "valid loss": loss_valid, "test loss": loss_test}

    return ep + 1, mean_err_train, mean_err_test, loss_dict

def LearnLMPlot(save_path, network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs, lr):
    ep, mean_err_train, mean_err_test, loss_dict = LearnLM(network, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs, lr)
    output.save_error_value(save_path, ep, mean_err_train, mean_err_test)
    output.save_plt(save_path, loss_dict, ep)
    return ep, mean_err_train, mean_err_test