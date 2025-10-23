---
description: ''
---

# 3. TRON1 实机部署

## 3.1 真机调试

1. 修改开发者电脑 IP：确保您的开发电脑与机器人本体通过外置网口连接。设置您的电脑 IP 地址为：`10.192.1.200`，并通过 Shell 命令`ping 10.192.1.2` 能够正常 ping 通。如下图所示对您的开发电脑进行 IP 设置：
   ![](./img/HxqWbMOXPoohgFxbuOOcmZTmnwc.png)
2. 打开开发者模式：机器人开机后，同时按下遥控器按键 `R1 + Left`，这时本体主机将会自动重启并切换机器人到开发者模式。在此模式下，用户可以开发自己的运动控制算法。模式设置掉电后不会失效，重新开机后仍然是开发者模式。
3. 进行校零动作：机器人开机启动后，执行运控程序之前，请进行校零，使机器人各个关节回到初始位置。校零对应的遥控器按键为 `L1 + R1`。
4. 实机部署运行。在 Bash 终端只需下面 Shell 命令启动控制算法（在进行实机部署运行时，<span style={{ backgroundColor: '#f65e59' }}>确保机器人吊装非常重要</span>）：
   ```bash
   conda activate pointfoot_deploy
   cd ~/limx_ws
   python rl-deploy-with-python/main.py 10.192.1.2
   ```

- 这时您可以通过遥控器按键`L1 + △`开启机器人行走功能。左摇杆：控制前进/后退/左转/右转运动；右遥控：可控制机器人左右横向运动。
- 遥控器按`L1 + □`关闭机器人行走功能。

## 3.2 真机部署

当您完成仿真和真机调试后，可以将您的算法程序部署到机器人上（在部署测试完成之前，确保机器人始终保持吊装状态非常重要，以保障安全）。详细步骤如下：

1. 准备工作
   - 保持机器人继续处于开发者模式：确保机器人仍处于开发者模式，方便您进行程序部署和调试。
   - 网络连接：确保开发电脑与机器人通过外置网口（Ethernet）连接，网络稳定且通信正常。部署完成后，网络连接将不再需要。
2. 拷贝算法程序到机器人

   - 在开发电脑上打开终端，进入到存放算法程序的工作目录，例如 `~/limx_ws`。
   - 使用 `scp` 命令将包含算法的目录拷贝到机器人中，默认机器人用户为 `guest`，密码为 `123456`。

     ```bash
      cd ~/limx_ws
      scp -r rl-deploy-with-python guest@10.192.1.2:/home/guest
     ```

   - 安装运动控制开发库（如果尚未安装）：
     ```bash
      ssh guest@10.192.1.2 "pip install /home/guest/rl-deploy-with-python/pointfoot-sdk-lowlevel/python3/amd64/limxsdk-\*-py3-none-any.whl"
     ```

- 配置算法自动启动：

  - SSH 登录机器人：使用 ssh 命令远程登录到机器人系统，密码为 `123456`。

    ```bash
    ssh guest@10.192.1.2mailto:guest@10.192.1.2
    ```

  - 打开自启动脚本 `/home/guest/autolaunch/autolaunch.sh` 进行编辑：

    ```bash
    busybox vi /home/guest/autolaunch/autolaunch.sh
    ```

  - 在脚本中找到启动 `main.py` 的命令：确保 `python3 /home/guest/rl-deploy-with-python/main.py 10.192.1.2` 这一行未被注释（没有 `#` 注释符号）。编辑完成后，保存并退出编辑器。

    ```bash
    #!/bin/bash

    while true; do
      # 启动用于控制 Pointfoot 机器人的 Python 控制器脚本
      # 参数 10.192.1.2 表示机器人所在的 IP 地址
      # 该行被注释掉，若要启用自动控制器启动，请取消注释
      # 请根据您控制器脚本的实际路径修改此路径
      python3 /home/guest/rl-deploy-with-python/main.py 10.192.1.2

      # 使用 roslaunch 启动机器人控制算法
      # 若要启动机器人控制算法，请取消下面注释
      # 请根据您安装的实际路径修改路径
      # source /home/guest/install/setup.bash
      # roslaunch robot_hw pointfoot_hw.launch

      # 等待 3 秒后重新启动
      sleep 3
    done
    ```

  - 重启机器人电脑。配置完成后，手动重启机器人。

3. 控制机器人运动  
   等待系统启动后，您可以使用遥控器控制机器人：
   - L1 + △：启动机器人行走功能；
   - L1 + □： 关闭机器人行走功能；
   - 左摇杆：控制机器人前进、后退、左转、右转；
   - 右摇杆：控制机器人左右横向运动。
