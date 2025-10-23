---
description: ''
---

# 1. 安装与环境配置

## 1.1 前置条件

1. GPU：推荐 GeForce RTX 3080 （12 GB） 或更高
2. 操作系统：Ubuntu 20.04 LTS
3. 网络环境：能正常访问 Github

## 1.2 一键安装 RL 训练环境

在终端中输入：

```bash
mkdir -p ~/limx_rl && cd ~/limx_rl && \
sudo apt update && sudo apt install -y git && \
if [ ! -d "pointfoot-legged-gym" ]; then \
  git clone https://github.com/limxdynamics/pointfoot-legged-gym.git; \
fi && \
cd pointfoot-legged-gym && bash install.sh && source ~/.bashrc
```

执行完上述命令后等待片刻，终端中会出现：
![](./img/PT5mbueSgokWdGxGlFfcRGHvnPb.png)
此时长按 Enter 阅读完 Anaconda 的用户协议后，会出现：
![](./img/Cl5Hb4RZEodjaSxNFuacrUrZn2g.png)
输入 yes 并回车后，可以选择 Anaconda 的路径（一般不用更改），直接按下 Enter 即可，会出现：
![](./img/ELnGbphzOoM2max8gYPcrk6Inr9.png)
输入 no 按下回车即可完成 Anaconda 的安装。
Anaconda 安装完成后，一键安装脚本还会继续安装 `pointfoot_legged_gym` 和相关依赖，请耐心等待安装完成。

## 1.3 验证训练环境

### 1.3.1 验证 Nvidia 驱动是否安装成功

```bash
nvidia-smi
```

如果显示了 NVIDIA 驱动版本和 GPU 信息，说明驱动已成功安装。
![](./img/BO2lbCJyaonHFhxmsyoc8stCnwh.png)

### 1.3.2 验证 Isaac Gym 是否安装成功

运行 Isaac Gym 的示例，它可以帮助确认 Isaac Gym 是否正确安装并配置。

```bash
conda activate pointfoot_legged_gym
cd ~/limx_rl/isaacgym/python/examples
python 1080_balls_of_solitude.py
```

![](./img/D3Q2bb15woi1P8xJMQDcpA4wneg.png)

### 1.3.3 验证 `pointfoot-legged-gym` 是否安装成功

当您安装完 RL 环境后，在 `~/limx_rl` 目录下包含如下所示的内容：

```bash
cd ~/limx_rl
.
├── isaacgym
└── pointfoot-legged-gym
```

## 1.4 一键安装 RL 部署环境

打开一个终端，在终端中输入（注意训练的 conda 环境和部署的 conda 环境不是同一个）：

```bash
source ~/anaconda3/etc/profile.d/conda.sh && \
conda create -n pointfoot_deploy python=3.8 -y && \
conda activate pointfoot_deploy && \
mkdir -p ~/limx_ws && cd ~/limx_ws && \
git clone --recurse-submodules https://github.com/limxdynamics/pointfoot-mujoco-sim.git && \
git clone --recurse-submodules https://github.com/limxdynamics/rl-deploy-with-python.git && \
ARCH=$(uname -m) && \
if [ "$ARCH" = "x86_64" ]; then \
  pip install rl-deploy-with-python/pointfoot-sdk-lowlevel/python3/amd64/limxsdk-*-py3-none-any.whl; \
elif [ "$ARCH" = "aarch64" ]; then \
  pip install rl-deploy-with-python/pointfoot-sdk-lowlevel/python3/aarch64/limxsdk-*-py3-none-any.whl; \
else \
  echo "不支持的架构：$ARCH"; \
fi && \
pip install onnx && \
echo 'export ROBOT_TYPE=PF_TRON1A' >> ~/.bashrc && source ~/.bashrc
```

## 1.5 验证部署环境

- 打开一个 Bash 终端来运行控制算法。

  ```bash
  # 激活pointfoot_deploy的conda环境
  conda activate pointfoot_deploy

  # 运行控制算法
  cd ~/limx_ws && python rl-deploy-with-python/main.py
  ```

- 打开一个 Bash 终端来运行 MuJoCo 仿真器。

  ```bash
  # 激活pointfoot_deploy的conda环境
  conda activate pointfoot_deploy

  # 运行仿真器
  cd ~/limx_ws && python pointfoot-mujoco-sim/simulator.py
  ```

- 打开一个 Bash 终端来运行虚拟遥控器，仿真的过程中，您可以使用虚拟遥控器来控制机器人运动

  - 左摇杆：控制前进/后退/左转/右转运动； 右摇杆：可控制机器人左右横向运动。

  ```bash
  cd ~/limx_ws && ./pointfoot-mujoco-sim/robot-joystick/robot-joystick
  ```

若能看到双点足机器人在 MuJoCo 中正常运动说明配置成功。

如果您的机器人处于摔倒状态，请在启动运动控制后，单击 MuJoCo 仿真器左侧菜单栏的 Reset 按钮以重置机器人。此时，您将看到机器人恢复并开始行走。

![](./img/Pb0kbpZMqosKQdx2j1ecxYa3njf.png)
