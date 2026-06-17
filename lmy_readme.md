<!-- bash-->
source devel_isolated/setup.bash
<!-- realsense -->
roslaunch realsense2_camera rs_camera.launch
<!-- init_ur -->
roslaunch Move_UR init_ur_robot.launch local_gripper_communicate:=False
<!-- init_gripper -->
sudo chmod 777 /dev/ttyUSB0
roslaunch Move_UR control_robotiq.launch local_gripper_communicate:=True
<!-- realsense calibration -->
<!--/home/yhx/shw_eyehand/src/easy_handeye/easy_handeye/launch/-->
cd /home/yhx/shw_eyehand
source devel/setup.bash
roslaunch realsense2_camera rs_camera.launch
roslaunch easy_handeye ur5_kinect_calibration_realsense_eye_to_hand.launch
roslaunch easy_handeye ur5_kinect_calibration_realsense_eye_on_hand.launch

translation: 
  x: 1.19167292469
  y: 0.0412377127321
  z: 0.448530014115
rotation: 
  x: -0.636075204934
  y: -0.624522710516
  z: 0.318418998797
  w: 0.322473346064

reboot robot need to reset robotiq
roslaunch Move_UR control_robotiq.launch local_gripper_communicate:=True

press: r

init robot()
source devel_isolated/setup.bash
roslaunch Move_UR init_ur_robot.launch  local_gripper_communicate:=True

move initial pose()
source devel_isolated/setup.bash
roslaunch Move_UR move_follow_action.launch

internet control(regnet)
source devel_isolated/setup.bash
python src/Move_UR/scripts/publish_action_client_network.py
