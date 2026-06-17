#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
SpaceMouse -> UR5e end-effector STEP teleoperation via the existing
/set_robot_action service (Move_UR/Action): absolute Cartesian pose targets.

Chain:
    spacenavd -> spacenav_node -> /spacenav/twist + /spacenav/joy ->
        this node --8-float [x,y,z,qx,qy,qz,qw,gripper]--> /set_robot_action
        -> URMoveServer.move_once_by_end()  (pose_based_cartesian_traj_controller)

Notes:
    * Each service call is BLOCKING (the server waits for the trajectory,
      run_time ~0.8 s in move_ur_follow_stepAction.py), so this is a discrete
      "step" teleop (~1 Hz), NOT continuous velocity. In exchange it avoids the
      speedl noise / force-limit / singularity blow-ups of the twist path.
    * The gripper is actuated SERVER-SIDE by the 8th element (1 = close, 0 = open),
      exactly like the existing publish_action_client_*.py. The
      move_ur_follow_stepAction server must be running.
    * Target pose is integrated from the CURRENT measured pose each step, so there
      is no drift (self-correcting from /tf):
        target_pos  = cur_pos  + lin_sign * deadzone(offset_lin) * max_step_lin
        target_quat = dq (x) cur_quat     (dq from ang_sign*deadzone(offset_ang)*max_step_ang)
    * Rotation frame: `~rot_frame` = "base" (dq on the left, world-fixed axes) or
      "tool" (dq on the right, tool-fixed axes).
"""

from __future__ import print_function

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from tf2_msgs.msg import TFMessage
from tf.transformations import quaternion_multiply, quaternion_from_euler
from Move_UR.srv import Action


class SpaceMouseActionTeleop(object):
    def __init__(self):
        rospy.init_node("spacemouse_action_teleop")

        # --- parameters ------------------------------------------------------
        self.max_step_lin = rospy.get_param("~max_step_lin", 0.02)   # m   per full-deflection step
        self.max_step_ang = rospy.get_param("~max_step_ang", 0.10)   # rad per full-deflection step
        self.deadzone = rospy.get_param("~deadzone", 0.2)
        self.input_scale = rospy.get_param("~input_scale", 1.0)
        self.poll_hz = rospy.get_param("~poll_rate", 20.0)
        self.settle = rospy.get_param("~settle_time", 0.05)          # s, let /tf update after a move
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
        self.debug = rospy.get_param("~debug", False)

        # --- state -----------------------------------------------------------
        self.ee_pose = None                       # [x,y,z,qx,qy,qz,qw]
        self.offset = np.zeros(6, dtype=np.float64)
        self.last_offset_stamp = rospy.Time(0)
        self.buttons = []
        self.prev_gripper_pressed = 0
        self.gripper_closed = False
        self.gripper_dirty = False                # gripper toggled -> need a move-in-place

        # --- service ---------------------------------------------------------
        rospy.loginfo("waiting for /set_robot_action ...")
        rospy.wait_for_service("/set_robot_action")
        self.set_action = rospy.ServiceProxy("/set_robot_action", Action)
        rospy.loginfo("/set_robot_action connected")

        # --- io --------------------------------------------------------------
        rospy.Subscriber("/tf", TFMessage, self.tf_cb, queue_size=10)
        rospy.Subscriber("/spacenav/twist", Twist, self.twist_cb, queue_size=1)
        rospy.Subscriber("/spacenav/joy", Joy, self.joy_cb, queue_size=1)

        rospy.loginfo("spacemouse_action_teleop ready: step_lin=%.3f m, step_ang=%.3f rad, "
                      "deadman=%s(btn %d), rot_frame=%s",
                      self.max_step_lin, self.max_step_ang,
                      self.require_deadman, self.deadman_button, self.rot_frame)

    # ----------------------------------------------------------------- helpers
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
            self.gripper_closed = not self.gripper_closed
            self.gripper_dirty = True
            rospy.loginfo("gripper -> %s", "CLOSE" if self.gripper_closed else "OPEN")
        self.prev_gripper_pressed = pressed

    # ---------------------------------------------------------------------- run
    def run(self):
        rate = rospy.Rate(self.poll_hz)
        while not rospy.is_shutdown():
            enabled = (not self.require_deadman) or self._button(self.deadman_button)
            fresh = (rospy.Time.now() - self.last_offset_stamp).to_sec() < self.cmd_timeout
            if enabled and fresh:
                off = self._apply_deadzone(np.clip(self.offset * self.input_scale, -1.0, 1.0))
            else:
                off = np.zeros(6)

            moving = bool(np.any(np.abs(off) > 0.0))
            if self.ee_pose is None:
                rate.sleep()
                continue
            if not (moving or self.gripper_dirty):
                rate.sleep()
                continue

            cur = self.ee_pose.copy()
            # --- translation (base frame) ---
            d_lin = off[:3] * self.lin_sign * self.max_step_lin
            if self.lock_translation:
                d_lin[:] = 0.0
            target_pos = cur[:3] + d_lin
            # --- rotation ---
            d_ang = off[3:] * self.ang_sign * self.max_step_ang
            if self.lock_rotation:
                d_ang[:] = 0.0
            dq = quaternion_from_euler(d_ang[0], d_ang[1], d_ang[2])  # [x,y,z,w]
            cur_q = cur[3:7]
            if self.rot_frame == "tool":
                target_q = quaternion_multiply(cur_q, dq)
            else:
                target_q = quaternion_multiply(dq, cur_q)
            target_q = target_q / np.linalg.norm(target_q)

            grip = 1.0 if self.gripper_closed else 0.0
            action = np.concatenate([target_pos, target_q, [grip]]).tolist()

            if self.debug:
                rospy.loginfo("enabled=%s fresh=%s buttons=%s d_lin=%s d_ang=%s grip=%.0f",
                              enabled, fresh, self.buttons,
                              np.round(d_lin, 4).tolist(), np.round(d_ang, 4).tolist(), grip)
            try:
                self.set_action(action)            # BLOCKING (~0.8 s server-side)
            except rospy.ServiceException as e:
                rospy.logwarn("set_robot_action failed: %s", e)

            self.gripper_dirty = False
            if self.settle > 0:
                rospy.sleep(self.settle)            # let /tf reflect the new pose
            rate.sleep()


if __name__ == "__main__":
    SpaceMouseActionTeleop().run()
