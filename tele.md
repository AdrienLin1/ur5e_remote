# SpaceMouse → UR5e 笛卡尔速度遥操（twist_controller）

3Dconnexion SpaceMouse 通过 `spacenavd + spacenav_node` 接入 ROS1，
节点把 6DoF 摇杆映射成 `geometry_msgs/Twist`，发到 UR 驱动的 `twist_controller`，
实现丝滑的末端速度遥操；两个按键分别做 **deadman（按住才动）** 和 **夹爪开合**。

```
spacenavd(daemon) --libspnav--> spacenav_node --/spacenav/twist + /spacenav/joy-->
    spacemouse_twist_teleop.py --/twist_controller/command--> ur_robot_driver(speedl)
```

---

## 1. 安装 spacenavd + spnav + spacenav_node

`$ROS_DISTRO` 用你的发行版（Melodic→`melodic`，Noetic→`noetic`）。

```bash
# (a) 系统层：守护进程 + libspnav（spnav 的底层库）
sudo apt update
sudo apt install -y spacenavd libspnav-dev

# 启动并设为开机自启
sudo systemctl enable spacenavd
sudo systemctl start  spacenavd
systemctl status spacenavd        # 看到 active (running) 即可

# (b) ROS 桥接节点：把设备发布成 /spacenav/twist 和 /spacenav/joy
sudo apt install -y ros-melodic-spacenav-node
```

> 说明：`spacenavd` 以 root 守护进程方式直接读 `/dev/input/*`，并通过
> `/var/run/spnav.sock` 把数据给 libspnav 客户端（`spacenav_node` 就是其中之一），
> 所以**通常不用配 udev 权限**。若 `spacenavd` 起不来，检查设备是否插好、
> 把当前用户加入 `input` 组：`sudo usermod -aG input $USER`（需重新登录）。

### 验证设备
```bash
lsusb | grep -iE '3dconnexion|logitech'   # 能看到设备

roscore &
rosrun spacenav_node spacenav_node
# 另开终端：
rostopic echo /spacenav/twist   # 推/扭摇杆，linear/angular 应有数值
rostopic echo /spacenav/joy     # 按左右键，buttons[0]/[1] 应在 0/1 跳变
```
记下你设备上 **哪个按键对应 buttons[0]、哪个对应 buttons[1]**，后面要用。

### （可选）python `spnav` 库
只有当你想直接复用 `roby` 里的 `SpacemouseTeleop` 类时才需要；本方案走
`spacenav_node`，**不需要**这个库。如需：`pip install spnav`（py2；py3 需用
社区 fork），并确保上面的 `spacenavd` 守护进程在跑。

---

## 2. 确认 twist_controller 可用

`twist_controller` 已经写在驱动配置里
（`ur_robot_driver/config/ur5e_controllers.yaml`，名字就叫 `twist_controller`，
命令话题 `/twist_controller/command`，类型 `geometry_msgs/Twist`）。
只需确保对应包已编译、controller_manager 能加载：

```bash
# 工作空间已编译（twist_controller / ros_controllers_cartesian 在你的仓库里）
cd ~/<your_catkin_ws> && catkin_make   # 或 catkin build
source devel/setup.bash

# 启动 UR 驱动后，列出可加载的控制器，应能看到 twist_controller
rosrun controller_manager controller_manager list
```
节点启动时会自动 `load + switch` 到 `twist_controller`（停掉冲突的轨迹控制器），
退出时切回 `scaled_pos_joint_traj_controller`，无需手动切换。

---

## 3. 运行

```bash
# 0) 机械臂驱动（含 moveit/相机按需）；务必在示教器上播放 External Control 程序
roslaunch Move_UR init_ur_robot.launch robot_ip:=192.168.1.138

# 1) 给节点加可执行权限（首次）
chmod +x $(rospack find Move_UR)/scripts/spacemouse_twist_teleop.py

# 2) 一键启动 spacenav_node + 遥操节点
roslaunch Move_UR spacemouse_twist_teleop.launch
#   常用可调参数：
#   roslaunch Move_UR spacemouse_twist_teleop.launch max_lin_vel:=0.15 max_ang_vel:=0.5 use_gripper:=false
```

操作：
- **按住 deadman（默认左键 buttons[0]）** 时，机械臂才跟随摇杆；松开立即停。
- **夹爪键（默认右键 buttons[1]）** 按一下切换开/合。
- 松开摇杆回中 → 速度归零（`spacenavd` 的 `zero_when_static`）。

---

## 4. 现场标定（重要）

twist 是在**机器人基座坐标系**下解释的，SpaceMouse 的安放方向决定了轴向对应关系，
**首次必须低速标定**：

1. 先把速度调到很低：`max_lin_vel:=0.05 max_ang_vel:=0.2`，手放在急停上。
2. 沿一个方向推摇杆，看末端往哪动。如果方向是反的，翻转对应分量的符号——
   改 launch 里的 `lin_sign`/`ang_sign`（顺序 `[x, y, z]`，`1.0`/`-1.0`）。
   例如左右推但机械臂前后动，说明你需要交换/翻转轴（先翻符号，若是轴序错了再
   在节点 `_shape()` 里加重排）。
3. 旋转同理，用 `ang_sign` 标定。
4. 标好后再逐步加大 `max_lin_vel` / `max_ang_vel`。

> 参考：`roby` 的 `action_mapping` 对 Franka 用的是平移/旋转各 `[-1, 1, -1]`，
> 可作为符号起点，但 UR 基座系和设备安放不同，仍以实测为准。

---

## 5. 参数速查（`spacemouse_twist_teleop.py`）

| 参数 | 默认 | 含义 |
|---|---|---|
| `~rate` | 125 | 向 twist_controller 发指令的频率 (Hz) |
| `~max_lin_vel` | 0.25 | 满偏时**实际**线速度 (m/s) |
| `~max_ang_vel` | 0.8 | 满偏时**实际**角速度 (rad/s) |
| `~controller_gain` | 0.1 | 必须等于 twist_controller 的 `twist_gain`，节点据此预除补偿 |
| `~deadzone` | 0.15 | 归一化输入上的死区 |
| `~input_scale` | 1.0 | 把 `/spacenav/twist` 归一化到约 ±1 的缩放 |
| `~cmd_timeout` | 0.3 | 超过此秒数没收到摇杆数据→发零 (s) |
| `~require_deadman` | true | 是否必须按住 deadman 才动 |
| `~deadman_button` | 0 | deadman 按键索引（/spacenav/joy 的 buttons） |
| `~gripper_button` | 1 | 夹爪切换按键索引 |
| `~use_gripper` | true | 是否驱动 Robotiq 夹爪 |
| `~lin_sign` / `~ang_sign` | [1,1,1] | 每轴符号标定 |
| `~lock_translation` / `~lock_rotation` | false | 只调姿态 / 只调位置时锁另一半 |
| `~manage_controller` | true | 启动自动切到 twist_controller，退出切回 |
| `~restore_controller` | scaled_pos_joint_traj_controller | 退出时切回的控制器 |

---

## 6. 安全须知

- `twist_controller` **没有看门狗**：最后一条非零 twist 会持续执行。本节点已做
  “持续发零 / 松手发零 / 退出发零”三重保护，但**急停必须随时可按**。
- 不要用 `kill -9` 杀节点（会跳过 `on_shutdown` 的归零）。用 Ctrl-C。
- 机械臂必须处于 remote control，且示教器上 External Control 程序正在运行，
  否则 speedl 指令不会生效。
- 录数据可继续用本仓库的 `Auto_Run_Collection`（`collect_ur_data.py`），它只订阅
  `/tf`、`/joint_states`、夹爪话题，和本遥操方式无关，可并行使用。
