#!/usr/bin/env python3
"""第03次课完整实验：只在CPU上运行参数估计、autograd、optimizer和validation。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dlwpt-lesson03-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dlwpt-lesson03-cache")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
import torch.optim as optim

SEED = 20260902
DEVICE = torch.device("cpu")
torch.manual_seed(SEED)
torch.set_printoptions(edgeitems=2, linewidth=88, precision=6)

# 目标温度与未知温标读数。所有Tensor均显式留在CPU。
t_c = torch.tensor(
    [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0],
    dtype=torch.float32,
    device=DEVICE,
)
t_u = torch.tensor(
    [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4],
    dtype=torch.float32,
    device=DEVICE,
)
t_un = 0.1 * t_u


def model(inputs: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """线性model：预测值等于w乘输入再加b。"""
    return w * inputs + b


def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error。"""
    squared_diffs = (predictions - targets) ** 2
    return squared_diffs.mean()


def dloss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE关于每个prediction的导数。"""
    return 2 * (predictions - targets) / predictions.size(0)


def dmodel_dw(inputs: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    del w, b
    return inputs


def dmodel_db(inputs: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> float:
    del inputs, w, b
    return 1.0


def grad_fn(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """用chain rule手工计算loss关于w和b的gradient。"""
    dloss_dtp = dloss_fn(predictions, targets)
    dloss_dw = dloss_dtp * dmodel_dw(inputs, w, b)
    dloss_db = dloss_dtp * dmodel_db(inputs, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


def manual_training_loop(
    n_epochs: int,
    learning_rate: float,
    params: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    print_params: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """使用手工gradient更新w与b。"""
    params = params.clone().to(DEVICE)
    history: list[float] = []
    report_epochs = {1, 2, 3, 10, 11, 99, 100, 4000, 5000}

    for epoch in range(1, n_epochs + 1):
        w, b = params
        predictions = model(inputs, w, b)
        loss = loss_fn(predictions, targets)
        grad = grad_fn(inputs, targets, predictions, w, b)
        params = params - learning_rate * grad
        history.append(loss.detach().item())

        if epoch in report_epochs:
            print(f"  Epoch {epoch:4d} | loss={loss.item():12.6f}")
            if print_params:
                print(f"             params={params} | grad={grad}")
        if not torch.isfinite(loss).all():
            print(f"  在epoch={epoch}检测到非有限loss，停止该组对照。")
            break

    return params, history


def autograd_training_loop(
    n_epochs: int,
    learning_rate: float,
    params: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    """由autograd求gradient，但仍手工执行parameter update。"""
    params = params.clone().detach().to(DEVICE).requires_grad_()
    history: list[float] = []

    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()

        predictions = model(inputs, *params)
        loss = loss_fn(predictions, targets)
        loss.backward()

        with torch.no_grad():
            params -= learning_rate * params.grad

        history.append(loss.detach().item())
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d} | loss={loss.item():10.6f}")

    return params.detach(), history


def optimizer_training_loop(
    n_epochs: int,
    optimizer: optim.Optimizer,
    params: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    """完整的zero_grad→forward→loss→backward→step闭环。"""
    history: list[float] = []

    for epoch in range(1, n_epochs + 1):
        predictions = model(inputs, *params)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.detach().item())
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d} | loss={loss.item():10.6f}")

    return params.detach().clone(), history


def train_validation_loop(
    n_epochs: int,
    optimizer: optim.Optimizer,
    params: torch.Tensor,
    train_inputs: torch.Tensor,
    val_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    use_no_grad: bool,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """只用training loss更新parameters，同时记录independent validation loss。"""
    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(1, n_epochs + 1):
        train_predictions = model(train_inputs, *params)
        train_loss = loss_fn(train_predictions, train_targets)

        if use_no_grad:
            with torch.no_grad():
                val_predictions = model(val_inputs, *params)
                val_loss = loss_fn(val_predictions, val_targets)
                assert val_loss.requires_grad is False
        else:
            val_predictions = model(val_inputs, *params)
            val_loss = loss_fn(val_predictions, val_targets)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_history.append(train_loss.detach().item())
        val_history.append(val_loss.detach().item())
        if epoch <= 3 or epoch % 500 == 0:
            print(
                f"  Epoch {epoch:4d} | training loss={train_loss.item():10.6f}"
                f" | validation loss={val_loss.item():10.6f}"
            )

    return params.detach().clone(), train_history, val_history


def save_data_plot() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.set_xlabel("Unknown measurement")
    ax.set_ylabel("Target temperature (°C)")
    ax.scatter(t_u.numpy(), t_c.numpy(), label="observations")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_temperature_data.png")
    plt.close(fig)


def save_fit_plot(params: torch.Tensor) -> None:
    predictions = model(t_un, *params)
    order = torch.argsort(t_u)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.set_xlabel("Unknown measurement")
    ax.set_ylabel("Target temperature (°C)")
    ax.plot(
        t_u[order].numpy(),
        predictions[order].detach().numpy(),
        label="fitted linear model",
    )
    ax.scatter(t_u.numpy(), t_c.numpy(), label="observations")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_manual_linear_fit.png")
    plt.close(fig)


def save_learning_rate_plot(
    divergent_history: list[float],
    slow_history: list[float],
    normalized_history: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    for values, label in (
        (divergent_history, "raw input, lr=1e-2"),
        (slow_history, "raw input, lr=1e-4"),
        (normalized_history, "normalized input, lr=1e-2"),
    ):
        finite_values = [value for value in values if torch.isfinite(torch.tensor(value))]
        if finite_values:
            ax.plot(range(1, len(finite_values) + 1), finite_values, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_learning_rate_comparison.png")
    plt.close(fig)


def save_train_validation_plot(train_history: list[float], val_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.plot(train_history, label="training loss")
    ax.plot(val_history, label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_train_validation_loss.png")
    plt.close(fig)


def experiment_1_parameter_estimation() -> torch.Tensor:
    print("\n=== 实验1：parameter estimation、loss与手工gradient ===")
    w = torch.ones((), device=DEVICE)
    b = torch.zeros((), device=DEVICE)
    predictions = model(t_u, w, b)
    initial_loss = loss_fn(predictions, t_c)
    print("初始predictions:", predictions)
    print(f"初始MSE: {initial_loss.item():.6f}")

    # Broadcasting实验：保留四种不同rank的Tensor组合。
    x = torch.ones((), device=DEVICE)
    y = torch.ones(3, 1, device=DEVICE)
    z = torch.ones(1, 3, device=DEVICE)
    a = torch.ones(2, 1, 1, device=DEVICE)
    print(f"broadcasting shapes: x={x.shape}, y={y.shape}, z={z.shape}, a={a.shape}")
    print("x * y shape:", (x * y).shape)
    print("y * z shape:", (y * z).shape)
    print("y * z * a shape:", (y * z * a).shape)

    # 中心有限差分估计loss关于w和b的变化率。
    delta = 0.1
    loss_rate_of_change_w = (
        loss_fn(model(t_u, w + delta, b), t_c)
        - loss_fn(model(t_u, w - delta, b), t_c)
    ) / (2.0 * delta)
    learning_rate = 1e-2
    w = w - learning_rate * loss_rate_of_change_w
    loss_rate_of_change_b = (
        loss_fn(model(t_u, w, b + delta), t_c)
        - loss_fn(model(t_u, w, b - delta), t_c)
    ) / (2.0 * delta)
    b = b - learning_rate * loss_rate_of_change_b
    print(
        "有限差分：",
        f"dL/dw={loss_rate_of_change_w.item():.6f},",
        f"dL/db={loss_rate_of_change_b.item():.6f},",
        f"updated w={w.item():.6f}, b={b.item():.6f}",
    )

    analytic_grad = grad_fn(t_u, t_c, model(t_u, w, b), w, b)
    print("解析gradient:", analytic_grad)

    print("\n[对照A] raw input + learning_rate=1e-2：预期发散")
    _, divergent_history = manual_training_loop(
        100, 1e-2, torch.tensor([1.0, 0.0]), t_u, t_c
    )

    print("\n[对照B] raw input + learning_rate=1e-4：稳定但较慢")
    _, slow_history = manual_training_loop(
        100, 1e-4, torch.tensor([1.0, 0.0]), t_u, t_c
    )

    print("\n[对照C] normalized input + learning_rate=1e-2：稳定且更快")
    _, normalized_history = manual_training_loop(
        100, 1e-2, torch.tensor([1.0, 0.0]), t_un, t_c
    )

    print("\n[完整拟合] normalized input，运行5000 epochs")
    fitted_params, _ = manual_training_loop(
        5000,
        1e-2,
        torch.tensor([1.0, 0.0]),
        t_un,
        t_c,
        print_params=True,
    )
    final_loss = loss_fn(model(t_un, *fitted_params), t_c)
    print(f"手工gradient最终params={fitted_params}, loss={final_loss.item():.6f}")

    save_data_plot()
    save_fit_plot(fitted_params)
    save_learning_rate_plot(divergent_history, slow_history, normalized_history)
    return fitted_params


def experiment_2_autograd() -> torch.Tensor:
    print("\n=== 实验2：autograd与手工parameter update ===")
    params = torch.tensor([1.0, 0.0], device=DEVICE, requires_grad=True)
    print("backward前params.grad is None:", params.grad is None)
    loss = loss_fn(model(t_u, *params), t_c)
    loss.backward()
    print("第一次backward后的params.grad:", params.grad)
    if params.grad is not None:
        params.grad.zero_()
    print("zero_后的params.grad:", params.grad)

    fitted_params, _ = autograd_training_loop(
        5000,
        1e-2,
        torch.tensor([1.0, 0.0], device=DEVICE, requires_grad=True),
        t_un,
        t_c,
    )
    final_loss = loss_fn(model(t_un, *fitted_params), t_c)
    print(f"autograd最终params={fitted_params}, loss={final_loss.item():.6f}")
    return fitted_params


def experiment_3_optimizers_and_validation() -> None:
    print("\n=== 实验3：optimizer、training/validation与gradient开关 ===")
    optimizer_names = sorted(
        name
        for name in dir(optim)
        if name[:1].isupper() and isinstance(getattr(optim, name), type)
    )
    print("torch.optim中可见的optimizer类（节选）:", optimizer_names[:12])

    # 单次SGD step：观察parameter如何发生变化。
    params = torch.tensor([1.0, 0.0], device=DEVICE, requires_grad=True)
    optimizer = optim.SGD([params], lr=1e-5)
    loss = loss_fn(model(t_u, *params), t_c)
    loss.backward()
    optimizer.step()
    print("raw input单次SGD step后的params:", params.detach())

    # 标准的zero_grad→backward→step顺序。
    params = torch.tensor([1.0, 0.0], device=DEVICE, requires_grad=True)
    optimizer = optim.SGD([params], lr=1e-2)
    loss = loss_fn(model(t_un, *params), t_c)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("normalized input单次标准SGD step后的params:", params.detach())

    print("\n[SGD完整拟合] normalized input，5000 epochs")
    params = torch.tensor([1.0, 0.0], device=DEVICE, requires_grad=True)
    optimizer = optim.SGD([params], lr=1e-2)
    sgd_params, _ = optimizer_training_loop(5000, optimizer, params, t_un, t_c)
    print("SGD最终params:", sgd_params)

    print("\n[Adam完整拟合] raw input，2000 epochs")
    params = torch.tensor([1.0, 0.0], device=DEVICE, requires_grad=True)
    optimizer = optim.Adam([params], lr=1e-1)
    adam_params, _ = optimizer_training_loop(2000, optimizer, params, t_u, t_c)
    print("Adam最终params:", adam_params)

    # 固定随机种子后划分independent training/validation samples。
    n_samples = t_u.shape[0]
    n_val = int(0.2 * n_samples)
    shuffled_indices = torch.randperm(n_samples, device=DEVICE)
    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]
    print("train_indices:", train_indices)
    print("val_indices:", val_indices)

    train_t_u = t_u[train_indices]
    train_t_c = t_c[train_indices]
    val_t_u = t_u[val_indices]
    val_t_c = t_c[val_indices]
    train_t_un = 0.1 * train_t_u
    val_t_un = 0.1 * val_t_u

    print("\n[train/val实验A] validation forward仍建立graph")
    params = torch.tensor([1.0, 0.0], device=DEVICE, requires_grad=True)
    optimizer = optim.SGD([params], lr=1e-2)
    _, _, _ = train_validation_loop(
        3000,
        optimizer,
        params,
        train_t_un,
        val_t_un,
        train_t_c,
        val_t_c,
        use_no_grad=False,
    )

    print("\n[train/val实验B] 使用torch.no_grad()执行validation forward")
    params = torch.tensor([1.0, 0.0], device=DEVICE, requires_grad=True)
    optimizer = optim.SGD([params], lr=1e-2)
    final_params, train_history, val_history = train_validation_loop(
        3000,
        optimizer,
        params,
        train_t_un,
        val_t_un,
        train_t_c,
        val_t_c,
        use_no_grad=True,
    )
    print("train/val最终params:", final_params)
    save_train_validation_plot(train_history, val_history)

    # set_grad_enabled适合把training与validation forward统一到一个函数中。
    params_for_forward = final_params.clone().detach().requires_grad_()

    def calc_forward(inputs: torch.Tensor, targets: torch.Tensor, is_train: bool) -> torch.Tensor:
        with torch.set_grad_enabled(is_train):
            predictions = model(inputs, *params_for_forward)
            return loss_fn(predictions, targets)

    train_forward_loss = calc_forward(train_t_un, train_t_c, is_train=True)
    val_forward_loss = calc_forward(val_t_un, val_t_c, is_train=False)
    print(
        "set_grad_enabled检查：",
        f"train requires_grad={train_forward_loss.requires_grad},",
        f"val requires_grad={val_forward_loss.requires_grad}",
    )
    assert train_forward_loss.requires_grad is True
    assert val_forward_loss.requires_grad is False


def main() -> None:
    assert DEVICE.type == "cpu"
    assert t_c.device.type == "cpu" and t_u.device.type == "cpu"
    print("PyTorch version:", torch.__version__)
    print("Execution device:", DEVICE)
    print("CUDA used: False")
    print("MPS used: False")

    manual_params = experiment_1_parameter_estimation()
    autograd_params = experiment_2_autograd()
    experiment_3_optimizers_and_validation()

    # 手工gradient与autograd应收敛到同一组参数附近。
    max_difference = (manual_params - autograd_params).abs().max().item()
    print(f"\nmanual vs. autograd parameter max difference: {max_difference:.8f}")
    assert max_difference < 1e-4
    print("所有实验完成，且全过程仅使用CPU。")
    print("输出图片目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
