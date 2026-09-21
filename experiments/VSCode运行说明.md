# 运行本实验

这是 Deep Learning with PyTorch 第 03 课实验：**学习机制（Mechanics of Learning）**。依赖只有 Python、PyTorch 和 matplotlib，全程使用 CPU，不需要 GPU。

有两种运行方式：

1. 本地 VS Code：`experiments/03_mechanics_of_learning_cpu.py`
2. Kaggle Jupyter Notebook：`03_mechanics_of_learning_kaggle.ipynb`

---

# 方法一：在本地 VS Code 中运行

## 1. 安装环境

- 安装 [Python 3.10+](https://www.python.org/downloads/)（安装时勾选 **Add python.exe to PATH**）
- 安装 [VS Code](https://code.visualstudio.com/)
- 在 VS Code 扩展里安装 **Python**（Microsoft）

## 2. 打开项目

`文件 → 打开文件夹`，选：

```text
C:\Users\yumen\Desktop\experiments
```

左侧应能看到 `experiments\03_mechanics_of_learning_cpu.py`。

## 3. 建虚拟环境并装依赖

在 VS Code 终端（`` Ctrl+` ``，选 PowerShell）执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch matplotlib
```

如果执行策略报错，先运行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后再激活 `.venv`。

## 4. 让 VS Code 用这个解释器

1. `Ctrl+Shift+P`
2. 选 **Python: Select Interpreter**
3. 选 `.venv\Scripts\python.exe`

打开 `03_mechanics_of_learning_cpu.py` 后，右下角应显示这个解释器。

## 5. 运行实验

任选一种：

- 打开该文件，按 **F5** 或点右上角 **Run Python File**
- 或在已激活 `.venv` 的终端里：

```powershell
python .\experiments\03_mechanics_of_learning_cpu.py
```

终端会打印三个实验的 loss / 参数。脚本使用 `Agg` 后端，**不会弹窗画图**，图会存到：

```text
C:\Users\yumen\Desktop\experiments\experiments\outputs\
```

应出现：

- `01_temperature_data.png`
- `02_manual_linear_fit.png`
- `03_learning_rate_comparison.png`
- `04_train_validation_loss.png`

成功结束时会看到类似：`所有实验完成，且全过程仅使用CPU。`

## Windows 上可能踩的坑

脚本里把 matplotlib 缓存指到了 Linux 路径 `/tmp/...`：

```python
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dlwpt-lesson03-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dlwpt-lesson03-cache")
```

如果报 matplotlib 配置目录错误，把这两行改成 Windows 临时目录，例如：

```python
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / "AppData" / "Local" / "Temp" / "dlwpt-lesson03-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.home() / "AppData" / "Local" / "Temp" / "dlwpt-lesson03-cache"))
```

之后重新运行即可。

---

# 方法二：在 Kaggle 上运行

Kaggle 已预装 PyTorch 和 matplotlib，免费 CPU 即可，不用本机装环境。

笔记本文件：`03_mechanics_of_learning_kaggle.ipynb`

## 1. 登录 Kaggle

打开 [https://www.kaggle.com](https://www.kaggle.com) 并登录。没有账号就先注册（可用 Google / GitHub 账号）。

## 2. 上传 Jupyter 笔记本

1. 点右上角 **Create** → **New Notebook**
2. 进入笔记本后，点 **File** → **Import Notebook**
3. 上传本仓库里的 `03_mechanics_of_learning_kaggle.ipynb`
4. 导入成功后，页面会打开这份实验笔记本

也可以先在本地打开该 `.ipynb`，全选复制，再粘贴到 Kaggle 新建的空白笔记本中。

## 3. 确认运行设置

在右侧 **Settings / Notebook options** 中：

- **Accelerator**：选 `None`（CPU 即可，本实验不需要 GPU）
- **Internet**：关掉即可，不需要额外装包
- **Language**：Python

## 4. 运行全部单元格

点顶部 **Run All**（或 `Shift+Enter` 逐格运行）。

终端/输出区会打印三个实验的 loss 和参数。图会直接显示在笔记本里，同时保存到：

```text
/kaggle/working/outputs/
```

应出现：

- `01_temperature_data.png`
- `02_manual_linear_fit.png`
- `03_learning_rate_comparison.png`
- `04_train_validation_loss.png`

成功结束时会看到类似：`所有实验完成，且全过程仅使用CPU。`

## 5. 下载结果（可选）

在右侧 **Output** / 工作目录里打开 `/kaggle/working/outputs/`，可逐张下载图片。也可以 **Save Version**，把这次运行结果保存在 Kaggle 上。

## 常见问题

- 如果提示找不到 `torch`：在第一个代码格最上面临时加一行 `%pip install torch matplotlib`，打开 Internet 后再 **Run All**。Kaggle 官方镜像一般已预装，通常不需要这一步。
- 本实验必须用 CPU。不要开 GPU，否则只是浪费配额，代码里也强制 `torch.device("cpu")`。
- 训练大约几分钟。若某个单元格正在跑，等它结束后再继续，不要重复点 Run All。
