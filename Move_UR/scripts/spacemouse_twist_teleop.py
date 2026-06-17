#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
SpaceMouse (3Dconnexion) -> UR5e Cartesian velocity teleoperation via twist_controller.

Pipeline:
    spacenavd (daemon) --libspnav--> spacenav_node --/spacenav/twist + /spacenav/joy-->
        this node --/twist_controller/command (geometry_msgs/Twist)--> ur_robot_driver (MODE_SPEEDL)

Notes / safety:
    * The UR `twist_controller` interprets the published Twist as TCP linear/angular
      velocity in the ROBOT BASE frame (driver sends it through URScript speedl()).
      The mapping between the SpaceMouse puck axes and the base frame depends on how
      the device is physically oriented w.r.t. the robot, so the per-axis signs
      (`~lin_sign`, `~ang_sign`) almost always need to be tuned live.
    * twist_controller has NO watchdog. The last non-zero command keeps executing.
      This node therefore (a) publishes a command every cycle (zero when idle),
      (b) publishes zero when the deadman is released or the input is stale, and
      (c) publishes zero on shutdown. Keep the e-stop in reach regardless.
    * The controller multiplies the incoming twist by `twist_gain` (default 0.1).
      We pre-divide by `~controller_gain` so that `~max_lin_vel` / `~max_ang_vel`
      are the ACTUAL end-effector velocities at full puck deflection.
"""

from __future__ import print_function

import sys
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from controller_manager_msgs.srv import (
    SwitchController, SwitchControllerRequest,
    LoadController, LoadControllerRequest,
    ListControllers, ListControllersRequest,
)

# Controllers that must not run at the same time as twist_controller.
JOINT_TRAJECTORY_CONTROLLERS = [
    "scaled_pos_joint_traj_controller",
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",
]
CARTESIAN_TRAJECTORY_CONTROLLERS = [
    "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller",
    "forward_cartesian_traj_controller",
]
TWIST_CONTROLLER = "twist_controller"


class SpaceMouseTwistTeleop(object):
    def __init__(self):
        rospy.init_node("spacemouse_twist_teleop")

        # --- parameters ------------------------------------------------------
        self.rate_hz = rospy.get_param("~rate", 125.0)
        # ACTUAL TCP velocity at full puck deflection
        self.max_lin = rospy.get_param("~max_lin_vel", 0.25)      # m/s
        self.max_ang = rospy.get_param("~max_ang_vel", 0.8)       # rad/s
        # twist_controller's internal twist_gain (see TwistController::twistCallback)
        self.controller_gain = rospy.get_param("~controller_gain", 0.1)
        self.deadzone = rospy.get_param("~deadzone", 0.15)        # on normalized [-1,1] input
        self.input_scale = rospy.get_param("~input_scale", 1.0)   # normalize /spacenav/twist to ~[-1,1]
        self.cmd_timeout = rospy.get_param("~cmd_timeout", 0.3)   # s, stale input -> zero
        self.require_deadman = rospy.get_param("~require_deadman", True)
        self.deadman_button = rospy.get_param("~deadman_button", 0)   # SpaceNav LEFT
        self.gripper_button = rospy.get_param("~gripper_button", 1)   # SpaceNav RIGHT
        self.use_gripper = rospy.get_param("~use_gripper", True)
        self.manage_controller = rospy.get_param("~manage_controller", True)
        self.restore_controller = rospy.get_param("~restore_controller",
                                                  "scaled_pos_joint_traj_controller")
        self.lock_translation = rospy.get_param("~lock_translation", False)
        self.lock_rotation = rospy.get_param("~lock_rotation", False)
        self.debug = rospy.get_param("~debug", False)
        # output smoothing: exponential moving average on the published twist.
        # 0 < alpha <= 1 ; lower = smoother/laggier, ramps up instead of stepping
        # (mitigates the jerk that trips the robot's force/protective stop and the noise).
        self.smoothing_alpha = float(rospy.get_param("~smoothing_alpha", 0.15))
        self.smoothing_alpha = min(max(self.smoothing_alpha, 0.01), 1.0)
        # per-axis sign / scale, order [x, y, z]; tune these to match your mounting
        self.lin_sign = np.array(rospy.get_param("~lin_sign", [1.0, 1.0, 1.0]), dtype=np.float64)
        self.ang_sign = np.array(rospy.get_param("~ang_sign", [1.0, 1.0, 1.0]), dtype=np.float64)

        if self.controller_gain <= 0.0:
            rospy.logwarn("controller_gain <= 0, forcing to 1.0")
            self.controller_gain = 1.0

        # --- state -----------------------------------------------------------
        self.latest_twist = np.zeros(6, dtype=np.float64)
        self.last_twist_stamp = rospy.Time(0)
        self.buttons = []
        self.prev_gripper_pressed = 0
        self.gripper_closed = False
        self.cmd_smooth = np.zeros(6, dtype=np.float64)  # smoothed output (pre-publish)

        # --- controller manager services ------------------------------------
        self.switch_srv = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
        self.load_srv = rospy.ServiceProxy("controller_manager/load_controller", LoadController)
        self.list_srv = rospy.ServiceProxy("controller_manager/list_controllers", ListControllers)
        if self.manage_controller:
            try:
                self.switch_srv.wait_for_service(5.0)
                self.switch_to(TWIST_CONTROLLER)
            except rospy.ROSException as err:
                rospy.logerr("Could not reach controller_manager: %s", err)
                sys.exit(-1)

        # --- gripper (optional) ---------------------------------------------
        # Reuse the SAME RobotiqGripper as useful_tool/control_robotiq.py so the
        # open/close behaviour matches your existing gripper control exactly
        # (rACT/rGTO/rSP/rFR + rPR=0 open / rPR=255 close on Robotiq2FGripperRobotOutput).
        self.gripper = None
        if self.use_gripper:
            try:
                from useful_tool.control_robotiq import RobotiqGripper
                self.gripper = RobotiqGripper(init_node=False)
                self.gripper.open_gripper()
                rospy.loginfo("gripper ready (toggle on button %d)", self.gripper_button)
            except Exception as e:  # noqa
                rospy.logwarn("gripper init failed (%s); running without gripper", e)
                self.gripper = None

        # --- pub / sub -------------------------------------------------------
        self.cmd_pub = rospy.Publisher("/twist_controller/command", Twist, queue_size=1)
        rospy.Subscriber("/spacenav/twist", Twist, self.twist_cb, queue_size=1)
        rospy.Subscriber("/spacenav/joy", Joy, self.joy_cb, queue_size=1)
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("spacemouse_twist_teleop ready: max_lin=%.3f m/s, max_ang=%.3f rad/s, "
                      "deadman=%s(btn %d)", self.max_lin, self.max_ang,
                      self.require_deadman, self.deadman_button)

    # ------------------------------------------------------------------ utils
    def switch_to(self, target):
        """Load `target` if needed and stop every other running controller."""
        resp = self.list_srv(ListControllersRequest())
        loaded = [c.name for c in resp.controller]
        for c in resp.controller:
            if c.name == target and c.state == "running":
                rospy.loginfo("%s already running", target)
                return
        if target not in loaded:
            rospy.loginfo("loading controller %s", target)
            self.load_srv(LoadControllerRequest(name=target))
        stop = [c.name for c in resp.controller
                if c.name != target and c.state == "running"]
        req = SwitchControllerRequest()
        req.start_controllers = [target]
        req.stop_controllers = stop
        req.strictness = SwitchControllerRequest.BEST_EFFORT
        rospy.loginfo("switching to %s (stopping %s)", target, stop)
        self.switch_srv(req)

    def _button(self, idx):
        if self.buttons is not None and len(self.buttons) > idx:
            return int(self.buttons[idx])
        return 0

    def _apply_deadzone(self, v):
        """Rescaled deadzone on a normalized [-1, 1] vector."""
        out = np.zeros_like(v)
        dz = self.deadzone
        mag = np.abs(v)
        active = mag > dz
        if dz >= 1.0:
            return out
        out[active] = np.sign(v[active]) * (mag[active] - dz) / (1.0 - dz)
        return np.clip(out, -1.0, 1.0)

    def _shape(self, raw):
        """raw: 6-vector from /spacenav/twist -> pre-gain Twist command (6-vector)."""
        norm = np.clip(raw * self.input_scale, -1.0, 1.0)
        lin = self._apply_deadzone(norm[:3]) * self.lin_sign * (self.max_lin / self.controller_gain)
        ang = self._apply_deadzone(norm[3:]) * self.ang_sign * (self.max_ang / self.controller_gain)
        if self.lock_translation:
            lin[:] = 0.0
        if self.lock_rotation:
            ang[:] = 0.0
        return np.concatenate([lin, ang])

    # ----------------------------------------------------------------- callbacks
    def twist_cb(self, msg):
        self.latest_twist = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z,
        ], dtype=np.float64)
        self.last_twist_stamp = rospy.Time.now()

    def joy_cb(self, msg):
        self.buttons = list(msg.buttons)
        pressed = self._button(self.gripper_button)
        if pressed and not self.prev_gripper_pressed:
            self.toggle_gripper()
        self.prev_gripper_pressed = pressed

    def toggle_gripper(self):
        if self.gripper is None:
            return
        if self.gripper_closed:
            self.gripper.open_gripper()
            self.gripper_closed = False
            rospy.loginfo("gripper -> OPEN")
        else:
            self.gripper.close_gripper()
            self.gripper_closed = True
            rospy.loginfo("gripper -> CLOSE")

    # ---------------------------------------------------------------------- run
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            twist = Twist()
            enabled = (not self.require_deadman) or self._button(self.deadman_button)
            age = (rospy.Time.now() - self.last_twist_stamp).to_sec()
            fresh = age < self.cmd_timeout
            if enabled and fresh:
                target = self._shape(self.latest_twist)
                # ramp toward target (smooth acceleration -> less force/jerk, less noise)
                self.cmd_smooth = (self.smoothing_alpha * target
                                   + (1.0 - self.smoothing_alpha) * self.cmd_smooth)
            else:
                # deadman released / stale input -> stop immediately (safety)
                self.cmd_smooth[:] = 0.0
            cmd = self.cmd_smooth
            twist.linear.x, twist.linear.y, twist.linear.z = cmd[0], cmd[1], cmd[2]
            twist.angular.x, twist.angular.y, twist.angular.z = cmd[3], cmd[4], cmd[5]
            if self.debug:
                rospy.loginfo_throttle(
                    0.5,
                    "enabled=%s(deadman btn%d=%d, require=%s) fresh=%s(age=%.2fs) "
                    "buttons=%s raw=%s -> out=[%.3f %.3f %.3f | %.3f %.3f %.3f]" % (
                        enabled, self.deadman_button, self._button(self.deadman_button),
                        self.require_deadman, fresh, age, self.buttons,
                        np.round(self.latest_twist, 3).tolist(),
                        twist.linear.x, twist.linear.y, twist.linear.z,
                        twist.angular.x, twist.angular.y, twist.angular.z))
            self.cmd_pub.publish(twist)  # zero Twist when idle / not enabled
            rate.sleep()

    def on_shutdown(self):
        rospy.loginfo("shutting down: sending zero twist")
        zero = Twist()
        for _ in range(10):
            self.cmd_pub.publish(zero)
            rospy.sleep(0.01)
        if self.manage_controller and self.restore_controller:
            try:
                self.switch_to(self.restore_controller)
            except Exception as e:  # noqa
                rospy.logwarn("failed to restore controller: %s", e)


if __name__ == "__main__":
    node = SpaceMouseTwistTeleop()
    node.run()
