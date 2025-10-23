---
description: ''
---

# 2. TRON1 点足训练

## 2.1 启动训练

1. 训练双点足机器人需切换分支，打开终端输入：

   ```bash
   cd ~/limx_rl/pointfoot-legged-gym && git checkout encoder-actor-critic
   ```

2. 常用训练相关参数指令说明：
   - --task=pointfoot_flat：指定任务或环境类型为 pointfoot_flat。
   - --num_envs 1024：指定要创建的环境数量为 1024。
   - --max_iteration 10000: 指定要训练的轮数为 10000 轮。
   - --headless：以无头模式运行，即不显示任何图形界面。通常用于在没有显示器的服务器上运行，或者在需要高效计算而不需要图形渲染的情况下使用。
3. 开始训练：

   1. 命令 1：无头模式运行（推荐使用）

      ```bash
      conda activate pointfoot_legged_gym
      cd ~/limx_rl/pointfoot-legged-gym
      python legged_gym/scripts/train.py --task=pointfoot_flat --headless
      ```

      此时训练以无图形化运行，由于不需要图形渲染，训练速度较快。

   2. 命令 2：图形模式运行

   ```bash
   conda activate pointfoot_legged_gym
   cd ~/limx_rl/pointfoot-legged-gym
   python legged_gym/scripts/train.py --task=pointfoot_flat
   ```

   可以看到 Isaac Gym 以图形模式运行：

   （可以在键盘按下 V 键暂停图形渲染来加快训练速度）
   ![](./img/Uc5ZbwGhroCa24xZykLc6Tninje.png)

## 2.2 继续训练：

- 如果您的训练终止，可以指定一个检查点的文件接着继续训练。
- 请注意将 `--load_run` 和 `--checkpoint` 的参数替换为实际中您的参数。

```bash
cd ~/limx_rl/pointfoot-legged-gym
python legged_gym/scripts/train.py --task=pointfoot_flat --resume --headless --load_run Dec23_17-38-22_ --checkpoint 200
```

可以看到 `Loading model from: ` 的输出，说明已经读取该路径下的 `checkpoint` 继续训练。
![](./img/RrFhbb4MloV4OgxKLjacQBhIndh.png)

## 2.3 查看训练情况

1. 激活 conda 环境

   ```bash
   conda activate pointfoot_legged_gym
   ```

2. 启动 tensorboard

   ```bash
   cd ~/limx_rl/pointfoot-legged-gym
   tensorboard --logdir=logs/pointfoot_flat
   ```

3. 查看训练情况

   在浏览器地址栏输入 [`http://127.0.0.1:6006`](http://127.0.0.1:6006)，查看训练情况。
   ![](./img/WDOeb7dRXokhyuxs4YAc3RV6n1e.png)

## 2.4 导出训练结果

1. 完成训练后查看训练结果
   默认会读取最新的 `run` 和 `checkpoint`，如需选择特定的 `run` 和 `checkpoint`，请输入 `--load_run` 和 `--checkpoint` 参数。

   - --load_run：指定要加载的训练运行的标识符（例如，训练运行的名字或 ID）。这个标识符通常与训练过程相关联，用于从 `logs` 目录中找到相应的运行记录或配置。
     - 获取方式：查看 `logs` 目录：进入 `logs` 目录查看子目录或文件，这些子目录或文件通常会以训练运行的标识符命名。
     - 示例路径：
     - ls -l ~/limx_rl/pointfoot-legged-gym/logs/pointfoot_flat
     - 您会看到类似于 `Dec23_17-38-22_` 这样的目录，`Dec23_17-38-22_` 就是 `--load_run` 参数的值。
       ![](./img/YyWpbEMG4oB36sx1bdhcOuqTnRc.png)
   - --checkpoint：指定要加载的检查点文件。检查点文件保存了模型的中间状态，可以用于恢复训练或进行推断。
     - 获取方式：查看 `logs` 目录：在 `logs` 目录下的相应 `--load_run` 目录中，通常会有保存检查点的文件。这些文件通常以 `.pt` 或类似扩展名存在，文件名可能包含训练的轮次或时间戳。
     - 示例路径：
     - ls -l ~/limx*rl/pointfoot-legged-gym/logs/pointfoot_flat/Dec23_17-38-22*
     - 您会看到类似于 `model_200.pt` 的文件，则 `200` 就是 `--checkpoint` 参数的值。
       ![](./img/Gv5Ybxeu9oA1E7xsUbocMWS8n3b.png)
   - 使用示例

     - 请注意将 `--load_run` 和 `--checkpoint` 后的参数替换为自己训练 `logs` 目录中的。

     ```bash
     conda activate pointfoot*legged_gym
     cd ~/limx_rl/pointfoot-legged-gym
     python legged_gym/scripts/play.py --task=pointfoot_flat --load_run Dec23_17-38-22* --checkpoint 200
     ```

     ![](./img/SSlCbCOb8oDh2OxRBFWcint7nff.png)

2. 导出模型的 ONNX 格式文件

   在运行完上一步的脚本后，可以到目录：  
   `~/limx_rl/pointfoot-legged-gym/logs/pointfoot_flat/exported/policies`
   中查看导出的文件。

   ```bash
   ls -l ~/limx_rl/pointfoot-legged-gym/logs/pointfoot_flat/exported/policies
   ```

   ![](./img/757a03a0-b050-452f-b541-44f45117fc64.png)

## 2.5 基于 Python 仿真部署（sim2sim）

1. 在您的工作空间中，以 `PF_TRON1A`机器人类型为例，RL 模型和配置文件所在路径为：
   `~/limx_ws/rl-deploy-with-python/controllers/model/PF_TRON1A`，如下所示。

   ```bash
   tree ~/limx_ws/rl-deploy-with-python/controllers/model/PF_TRON1A
   .
   ├── params.yaml
   └── policy
   └── policy.onnx
   └── encoder.onnx
   ```

2. 将导出的训练结果即 ONNX 文件（包括 `policy.onnx` 和 `encoder.onnx`）移动到对应的 policy 文件夹内即可。
3. 启动仿真

   - 打开一个 Bash 终端来运行控制算法。

     ```bash
     conda activate pointfoot_deploy
     cd ~/limx_ws && python rl-deploy-with-python/main.py
     ```

   - 打开一个 Bash 终端来运行 MuJoCo 仿真器。
     ```bash
     conda activate pointfoot_deploy
     cd ~/limx_ws && python pointfoot-mujoco-sim/simulator.py
     ```

- 打开一个 Bash 终端来运行虚拟遥控器。

  ```bash
  cd ~/limx_ws && ./pointfoot-mujoco-sim/robot-joystick/robot-joystick
  ```

即可看到训练结果部署在 MuJoCo 平台中的表现。
