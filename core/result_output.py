import os
import matplotlib.pyplot as plt

def save_error_value(save_path, ep, mean_err_train, mean_err_test, approximated=False):
    os.makedirs(save_path, exist_ok=True)
    if approximated:
        err_path = os.path.join(save_path, "mean_errors(approximated).txt")
    else:
        err_path = os.path.join(save_path, "mean_errors.txt")
    with open(err_path, "w", encoding="utf-8") as f:
        f.write(f"epochs:\t{ep}\n")
        f.write(f"mean error (train):{mean_err_train:.5e}\n")
        f.write(f"mean error (test):{mean_err_test:.5e}\n")

def save_plt(save_path, loss_dict, ep):
    for lab, loss_list in loss_dict.items():
        plt.plot([ii for ii in range(ep)], loss_list, label = lab)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f"{save_path}/loss", dpi=300, bbox_inches='tight')
    plt.close()