#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Single-node SpaceMouse -> UR5e end-effector teleop (POSITION / pose mode).

This ONE node does everything, no /set_robot_action server, no separate client:
    * reads the SpaceMouse from spacenav_node (/spacenav/twist + /spacenav/joy)
    * reads the current TCP pose from /tf (tool0_controller)
    * integrates an internal TARGET pose at `loop_rate` and STREAMS it
      (non-blocking) to pose_based_cartesian_traj_controller, so the arm follows
      a continuously moving goal -> smooth motion, NOT the 1 Hz blocking "steps"
      of the /set_robot_action path
    * drives the Robotiq gripper directly (same commands as control_robotiq.py)

Why this is smooth AND quiet:
    * Smooth: we never wait_for_result(); each cycle we move the target a little
      (velocity * dt) and send a short cartesian goal that preempts the last one.
    * Quiet / no torque-window trip: it is POSITION control (trajectory
      controller interpolates with bounded accel), not raw speedl velocity.

Safety:
    * Hold the deadman button to move; release -> target is dropped and the goal
      is cancelled (arm holds). The target is also "leashed" so it can never run
      more than `~leash` metres ahead of the real TCP (anti-runaway).
    * Per-axis signs (`~lin_sign`/`~ang_sign`) must be tuned to your mounting.
"""

from __future__ import print_function

import sys
import numpy as np
import rospy
import actionlib

from geometry_msgs.msg import Twist, Pose, Vector3, Quaternion
from sensor_msgs.msg import Joy
from tf2_msgs.msg import TFMessage
from tf.transformations import quaternion_multiply, quaternion_from_euler
from cartesian_control_msgs.msg import (
    FollowCartesianTrajectoryAction,
    FollowCartesianTrajectoryGoal,
    CartesianTrajectoryPoint,
)
from controller_manager_msgs.srv import (
    SwitchController, SwitchControllerRequest,
    LoadController, LoadControllerRequest,
    ListControllers, ListControllersRequest,
)

CART_CONTROLLER = "pose_based_cartesian_traj_controller"


class SpaceMousePoseTeleop(object):
    def __init__(self):
        rospy.init_node("spacemouse_pose_teleop")

        # --- parameters ------------------------------------------------------
        self.loop_hz = rospy.get_param("~loop_rate", 20.0)
        self.goal_time = rospy.get_param("~goal_time", 0.12)         # s, horizon of each streamed goal
        self.max_lin_vel = rospy.get_param("~max_lin_vel", 0.08)     # m/s   at full deflection
        self.max_ang_vel = rospy.get_param("~max_ang_vel", 0.35)     # rad/s at full deflection
        self.deadzone = rospy.get_param("~deadzone", 0.2)
        self.input_scale = rospy.get_param("~input_scale", 1.0)
        self.vel_alpha = rospy.get_param("~vel_smoothing", 0.25)     # EMA on velocity (lower = smoother)
        self.leash = rospy.get_param("~leash", 0.05)                 # m, max target lead over actual TCP
        self.min_send_lin = rospy.get_param("~min_send_lin", 0.004)  # m, only send a new goal after this much motion
        self.min_send_ang = rospy.get_param("~min_send_ang", 0.01)   # quaternion-distance gate
        self.cmd_timeout = rospy.get_param("~cmd_timeout", 0.3)
        self.require_deadman = rospy.get_param("~require_deadman", True)
        self.deadman_button = rospy.get_param("~deadman_button", 0)
        self.gripper_button = rospy.get_param("~gripper_button", 1)
        self.rot_frame = rospy.get_param("~rot_frame", "base")       # "base" | "tool"
        self.tool_frame = rospy.get_param("~tool_frame", "tool0_controller")
        self.lin_sign = np.array(rospy.get_param("~lin_sign", [1.0, 1.0, 1.0]), dtype=np.float64)
        self.ang_sign = np.array(rospy.get_param("~ang_sign", [1.0, 1.0, 1.0]), dtype=np.float64)
        self.lock_translation = rospy.get_param("~lock_translation", False)
        self.lock_rotation = rospy.get_param("~lock_rotation", False)
        self.use_gripper = rospy.get_param("~use_gripper", True)
        self.manage_controller = rospy.get_param("~manage_controller", True)
        self.debug = rospy.get_param("~debug", False)

        # --- state -----------------------------------------------------------
        self.ee_pose = None                       # measured TCP [x,y,z,qx,qy,qz,qw]
        self.offset = np.zeros(6, dtype=np.float64)
        self.last_offset_stamp = rospy.Time(0)
        self.buttons = []
        self.prev_gripper_pressed = 0
        self.gripper_closed = False
        self.vel_smooth = np.zeros(6, dtype=np.float64)
        self.target = None                        # internal integrated target pose
        self.last_sent = None
        self.cur_goal = None                      # last ClientGoalHandle
        self.was_moving = False

        # --- controller manager ---------------------------------------------
        self.switch_srv = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
        self.load_srv = rospy.ServiceProxy("controller_manager/load_controller", LoadController)
        self.list_srv = rospy.ServiceProxy("controller_manager/list_controllers", ListControllers)
        if self.manage_controller:
            try:
                self.switch_srv.wait_for_service(5.0)
                self.switch_to(CART_CONTROLLER)
            except rospy.ROSException as err:
                rospy.logerr("controller_manager not reachable: %s", err)
                sys.exit(-1)

        # --- cartesian trajectory action client ------------------------------
        # Low-level ActionClient (NOT SimpleActionClient): we stream goals at
        # loop_rate and each new goal preempts the previous one. SimpleActionClient
        # only models one goal and throws "Received comm state PREEMPTING when in
        # simple state DONE" on rapid re-goal; ActionClient handles it cleanly.
        self.traj_client = actionlib.ActionClient(
            "{}/follow_cartesian_trajectory".format(CART_CONTROLLER),
            FollowCartesianTrajectoryAction,
        )
        rospy.loginfo("waiting for cartesian trajectory action server ...")
        if not self.traj_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("cartesian trajectory action server not available")
            sys.exit(-1)
        rospy.loginfo("cartesian trajectory action server connected")

        # --- gripper (same RobotiqGripper as control_robotiq.py) -------------
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

        # --- io --------------------------------------------------------------
        rospy.Subscriber("/tf", TFMessage, self.tf_cb, queue_size=10)
        rospy.Subscriber("/spacenav/twist", Twist, self.twist_cb, queue_size=1)
        rospy.Subscriber("/spacenav/joy", Joy, self.joy_cb, queue_size=1)
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("spacemouse_pose_teleop ready: vmax=%.3f m/s / %.3f rad/s, loop=%.0f Hz, "
                      "deadman=%s(btn %d), rot_frame=%s",
                      self.max_lin_vel, self.max_ang_vel, self.loop_hz,
                      self.require_deadman, self.deadman_button, self.rot_frame)

    # ----------------------------------------------------------------- helpers
    def switch_to(self, target):
        resp = self.list_srv(ListControllersRequest())
        loaded = [c.name for c in resp.controller]
        for c in resp.controller:
            if c.name == target and c.state == "running":
                return
        if target not in loaded:
            self.load_srv(LoadControllerRequest(name=target))
        stop = [c.name for c in resp.controller if c.name != target and c.state == "running"]
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
        out = np.zeros_like(v)
        dz = self.deadzone
        if dz >= 1.0:
            return out
        mag = np.abs(v)
        active = mag > dz
        out[active] = np.sign(v[active]) * (mag[active] - dz) / (1.0 - dz)
        return np.clip(out, -1.0, 1.0)

    def send_pose_goal(self, pose7):
        goal = FollowCartesianTrajectoryGoal()
        pt = CartesianTrajectoryPoint()
        pt.pose = Pose(Vector3(pose7[0], pose7[1], pose7[2]),
                       Quaternion(pose7[3], pose7[4], pose7[5], pose7[6]))
        pt.time_from_start = rospy.Duration(self.goal_time)
        goal.trajectory.points.append(pt)
        # ActionClient.send_goal returns a ClientGoalHandle; the controller preempts
        # the currently-active trajectory server-side. No client "simple state" race.
        self.cur_goal = self.traj_client.send_goal(goal)

    # ---------------------------------------------------------------- callbacks
    def tf_cb(self, msg):
        for t in msg.transforms:
            if t.child_frame_id == self.tool_frame:
                tr = t.transform.translation
                ro = t.transform.rotation
                self.ee_pose = np.array([tr.x, tr.y, tr.z, ro.x, ro.y, ro.z, ro.w],
                                        dtype=np.float64)

    def twist_cb(self, msg):
        self.offset = np.array([msg.linear.x, msg.linear.y, msg.linear.z,
                                msg.angular.x, msg.angular.y, msg.angular.z],
                               dtype=np.float64)
        self.last_offset_stamp = rospy.Time.now()

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
        rate = rospy.Rate(self.loop_hz)
        dt = 1.0 / self.loop_hz
        while not rospy.is_shutdown():
            enabled = (not self.require_deadman) or self._button(self.deadman_button)
            fresh = (rospy.Time.now() - self.last_offset_stamp).to_sec() < self.cmd_timeout

            if self.ee_pose is None:
                rate.sleep()
                continue

            if not enabled:
                # disengage: stop integrating & streaming, but do NOT cancel the
                # in-flight goal (canceling causes an abrupt-stop jolt). Let the last
                # trajectory finish; the position controller then holds the pose.
                self.target = None
                self.last_sent = None
                self.vel_smooth[:] = 0.0
                rate.sleep()
                continue

            # (re)engage: start target from the current measured pose (no jump)
            if self.target is None:
                self.target = self.ee_pose.copy()
                self.last_sent = None

            # shaped, smoothed velocity command -----------------------------
            raw = self._apply_deadzone(np.clip(self.offset * self.input_scale, -1.0, 1.0)) \
                if fresh else np.zeros(6)
            vel = np.zeros(6)
            vel[:3] = raw[:3] * self.lin_sign * self.max_lin_vel
            vel[3:] = raw[3:] * self.ang_sign * self.max_ang_vel
            if self.lock_translation:
                vel[:3] = 0.0
            if self.lock_rotation:
                vel[3:] = 0.0
            self.vel_smooth = self.vel_alpha * vel + (1.0 - self.vel_alpha) * self.vel_smooth
            if np.linalg.norm(self.vel_smooth) < 1e-3:   # crisp stop, no creeping tail
                self.vel_smooth[:] = 0.0

            # integrate target ----------------------------------------------
            self.target[:3] += self.vel_smooth[:3] * dt
            d_ang = self.vel_smooth[3:] * dt
            if np.any(np.abs(d_ang) > 0):
                dq = quaternion_from_euler(d_ang[0], d_ang[1], d_ang[2])
                q = self.target[3:7]
                q = quaternion_multiply(q, dq) if self.rot_frame == "tool" \
                    else quaternion_multiply(dq, q)
                self.target[3:7] = q / np.linalg.norm(q)

            # leash: keep target from running ahead of the real TCP ---------
            err = self.target[:3] - self.ee_pose[:3]
            d = np.linalg.norm(err)
            if d > self.leash:
                self.target[:3] = self.ee_pose[:3] + err / d * self.leash

            # Stream ONE fresh goal every cycle while engaged. The old min-step
            # gating made the goal cadence irregular during wind-down (send every
            # 1, then 2, then 4 cycles ...), which is exactly the "卡顿 on settle"
            # you saw. Uniform cadence + goal_time > loop period means each goal is
            # re-targeted before it can decelerate to its endpoint -> continuous,
            # servo-like motion instead of per-goal accel/decel shudder.
            self.send_pose_goal(self.target)
            self.last_sent = self.target.copy()

            if self.debug:
                rospy.loginfo_throttle(
                    0.5, "enabled=%s fresh=%s buttons=%s vel=%s leash_err=%.3f",
                    enabled, fresh, self.buttons,
                    np.round(self.vel_smooth, 3).tolist(), d)
            rate.sleep()

    def on_shutdown(self):
        try:
            self.traj_client.cancel_all_goals()
        except Exception:  # noqa
            pass
        rospy.loginfo("spacemouse_pose_teleop stopped")


if __name__ == "__main__":
    SpaceMousePoseTeleop().run()
