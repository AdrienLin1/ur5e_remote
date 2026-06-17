#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
SpaceMouse -> UR5e resolved-rate teleop via servoJ (forward_position_controller).

This is the ROS analog of the Quest example: a CONTINUOUS POSITION SERVO.
    * SpaceMouse puck            -> desired TCP twist V (base frame)
    * resolved-rate IK          -> dq = J(q)^+ V   (damped least squares, singular-safe)
    * integrate joint target    -> q_target += dq * dt
    * stream q_target on /forward_position_controller/command (Float64MultiArray)
    * the UR driver executes each target with servoJ (MODE_SERVOJ) at 500 Hz,
      using its own servoj_lookahead_time / servoj_gain to smooth between points.

Why this matches the Quest "丝滑":
    * servoJ is a real-time POSITION servo with built-in lookahead/gain blending.
      We feed it a fresh joint target every cycle; it never replans or decelerates
      to a trajectory endpoint, so there is no per-goal accel/decel shudder like the
      streamed FollowCartesianTrajectory path has.

Why it is quiet (unlike twist_controller / speedl):
    * It is POSITION control (servoJ), not speedl velocity mode, so there is no
      continuous speed-mode motor whine ("滋滋"). Between motions it simply holds
      the last joint target.

Controller:
    forward_position_controller (position_controllers/JointGroupPositionController).
    It is NOT in the driver's ur5e_controllers.yaml, so the launch file defines its
    params (type + joints) on the param server; controller_manager can then load it
    WITHOUT editing the driver config or restarting the driver. The driver maps the
    PositionJointInterface to servoJ (see hardware_interface.cpp, MODE_SERVOJ).

Safety:
    * Hold the deadman to move; release -> stop integrating (arm holds the last
      joint target). Re-engage re-seeds q_target from the MEASURED joints (no jump).
    * Per-cycle joint step is clamped (`~max_joint_speed`); damped least squares
      keeps dq bounded near singularities. Keep the e-stop in reach.
    * On shutdown we switch back to `~restore_controller` (default the scaled joint
      trajectory controller) and never publish a zero target (that would jump to 0).
"""

from __future__ import print_function

import sys
import numpy as np
import rospy

from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import (
    SwitchController, SwitchControllerRequest,
    LoadController, LoadControllerRequest,
    ListControllers, ListControllersRequest,
)

try:
    import PyKDL as kdl
    from kdl_parser_py.urdf import treeFromString
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "spacemouse_servoj_teleop needs PyKDL + kdl_parser_py. Install with:\n"
        "  sudo apt install ros-$ROS_DISTRO-python-orocos-kdl "
        "ros-$ROS_DISTRO-kdl-parser-py\n"
        "(original error: %s)" % exc)

# Controller joint order (also what we publish on the command topic).
UR_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
FWD_POS_CONTROLLER = "forward_position_controller"
# Everything that must be stopped before we can stream positions.
CONFLICTING = [
    "scaled_pos_joint_traj_controller", "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller", "vel_joint_traj_controller",
    "forward_joint_traj_controller", "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller", "forward_cartesian_traj_controller",
    "twist_controller",
]


class SpaceMouseServoJTeleop(object):
    def __init__(self):
        rospy.init_node("spacemouse_servoj_teleop")

        # --- parameters ------------------------------------------------------
        self.rate_hz = rospy.get_param("~rate", 125.0)
        self.max_lin = rospy.get_param("~max_lin_vel", 0.08)     # m/s at full deflection
        self.max_ang = rospy.get_param("~max_ang_vel", 0.35)     # rad/s at full deflection
        self.deadzone = rospy.get_param("~deadzone", 0.2)
        self.input_scale = rospy.get_param("~input_scale", 1.0)  # normalize /spacenav/twist to ~[-1,1]
        self.vel_alpha = rospy.get_param("~vel_smoothing", 0.25)  # EMA on the twist (lower = smoother)
        self.cmd_timeout = rospy.get_param("~cmd_timeout", 0.3)
        self.damping = rospy.get_param("~damping", 0.05)         # DLS lambda (rad/s per m/s ~ singular robustness)
        self.max_joint_speed = rospy.get_param("~max_joint_speed", 1.5)  # rad/s, per-joint clamp
        self.require_deadman = rospy.get_param("~require_deadman", True)
        self.deadman_button = rospy.get_param("~deadman_button", 0)
        self.gripper_button = rospy.get_param("~gripper_button", 1)
        self.use_gripper = rospy.get_param("~use_gripper", True)
        self.manage_controller = rospy.get_param("~manage_controller", True)
        self.restore_controller = rospy.get_param("~restore_controller",
                                                  "scaled_pos_joint_traj_controller")
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.tip_frame = rospy.get_param("~tip_frame", "tool0")
        self.lock_translation = rospy.get_param("~lock_translation", False)
        self.lock_rotation = rospy.get_param("~lock_rotation", False)
        self.lin_sign = np.array(rospy.get_param("~lin_sign", [1.0, 1.0, 1.0]), dtype=np.float64)
        self.ang_sign = np.array(rospy.get_param("~ang_sign", [1.0, 1.0, 1.0]), dtype=np.float64)
        self.debug = rospy.get_param("~debug", False)

        # --- KDL chain + Jacobian solver (from /robot_description) ------------
        ok, tree = self._load_kdl_tree()
        if not ok:
            rospy.logerr("failed to parse /robot_description (is the driver/MoveIt up?)")
            sys.exit(-1)
        self.chain = tree.getChain(self.base_frame, self.tip_frame)
        self.n = self.chain.getNrOfJoints()
        if self.n == 0:
            rospy.logerr("empty KDL chain %s -> %s; check base_frame/tip_frame",
                         self.base_frame, self.tip_frame)
            sys.exit(-1)
        # ordered movable-joint names in the chain (skip fixed joints, KDL type "None")
        kdl_names = [self.chain.getSegment(i).getJoint().getName()
                     for i in range(self.chain.getNrOfSegments())
                     if self.chain.getSegment(i).getJoint().getTypeName() != "None"]
        missing = [j for j in UR_JOINTS if j not in kdl_names]
        if missing:
            rospy.logerr("chain %s->%s is missing joints %s (got %s)",
                         self.base_frame, self.tip_frame, missing, kdl_names)
            sys.exit(-1)
        # perm[j] = column index in KDL order for UR_JOINTS[j]
        self.perm = [kdl_names.index(j) for j in UR_JOINTS]
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        rospy.loginfo("KDL chain %s -> %s (%d joints): %s",
                      self.base_frame, self.tip_frame, self.n, kdl_names)

        # --- state -----------------------------------------------------------
        self.q_meas = None                  # measured joints, UR_JOINTS order
        self._last_js_names = None          # for diagnostics if we never see UR joints
        self.q_target = None                # integrated joint target, UR_JOINTS order
        self.latest_twist = np.zeros(6, dtype=np.float64)
        self.last_twist_stamp = rospy.Time(0)
        self.vel_smooth = np.zeros(6, dtype=np.float64)
        self.buttons = []
        self.prev_gripper_pressed = 0
        self.gripper_closed = False

        # --- controller manager services ------------------------------------
        self.switch_srv = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
        self.load_srv = rospy.ServiceProxy("controller_manager/load_controller", LoadController)
        self.list_srv = rospy.ServiceProxy("controller_manager/list_controllers", ListControllers)

        # --- gripper (same RobotiqGripper as the other teleop nodes) ---------
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
        self.cmd_pub = rospy.Publisher("/{}/command".format(FWD_POS_CONTROLLER),
                                       Float64MultiArray, queue_size=1)
        rospy.Subscriber("/joint_states", JointState, self.js_cb, queue_size=1)
        rospy.Subscriber("/spacenav/twist", Twist, self.twist_cb, queue_size=1)
        rospy.Subscriber("/spacenav/joy", Joy, self.joy_cb, queue_size=1)
        rospy.on_shutdown(self.on_shutdown)

        # wait for the first joint state so the controller starts from a real pose
        rospy.loginfo("waiting for /joint_states carrying the UR joints ...")
        while self.q_meas is None and not rospy.is_shutdown():
            rospy.logwarn_throttle(
                2.0, "no UR joints in /joint_states yet. Need %s; last names seen = %s",
                UR_JOINTS, self._last_js_names)
            rospy.sleep(0.1)

        # only now switch the robot to position streaming (it starts holding the
        # measured pose, so there is no jump)
        if self.manage_controller and not rospy.is_shutdown():
            try:
                self.switch_srv.wait_for_service(5.0)
                self.switch_to(FWD_POS_CONTROLLER)
            except rospy.ROSException as err:
                rospy.logerr("controller_manager not reachable: %s", err)
                sys.exit(-1)

        rospy.loginfo("spacemouse_servoj_teleop ready: vmax=%.3f m/s / %.3f rad/s @ %.0f Hz, "
                      "deadman=%s(btn %d)", self.max_lin, self.max_ang, self.rate_hz,
                      self.require_deadman, self.deadman_button)

    # ----------------------------------------------------------------- helpers
    def _load_kdl_tree(self):
        """Build a KDL tree from /robot_description, robust to broken urdf_parser_py.

        Some workspaces shadow urdf_parser_py with an old fork (e.g. a Baxter SDK)
        that cannot duck-type-parse UR's <transmission> blocks
        ("Required element not set in XML: hardwareInterface"). KDL only needs the
        link/joint kinematics, so strip <transmission> and <gazebo> before parsing.
        """
        import xml.etree.ElementTree as ET
        urdf_xml = rospy.get_param("robot_description")
        try:
            root = ET.fromstring(urdf_xml)
            for tag in ("transmission", "gazebo"):
                for elem in root.findall(tag):
                    root.remove(elem)
            urdf_xml = ET.tostring(root)
        except Exception as e:  # noqa  - fall back to the raw URDF
            rospy.logwarn("could not strip transmission/gazebo (%s); trying raw URDF", e)
        return treeFromString(urdf_xml)

    def switch_to(self, target):
        """Load `target` if needed and stop every conflicting running controller."""
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
                if c.name in CONFLICTING and c.state == "running"]
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

    def _shape(self, raw):
        """raw 6-vector from /spacenav/twist -> desired TCP twist V (base frame)."""
        norm = np.clip(raw * self.input_scale, -1.0, 1.0)
        lin = self._apply_deadzone(norm[:3]) * self.lin_sign * self.max_lin
        ang = self._apply_deadzone(norm[3:]) * self.ang_sign * self.max_ang
        if self.lock_translation:
            lin[:] = 0.0
        if self.lock_rotation:
            ang[:] = 0.0
        return np.concatenate([lin, ang])

    def _jacobian(self, q_ur):
        """Geometric Jacobian (6 x n) at q, columns reordered to UR_JOINTS order,
        expressed in the chain base frame with reference point at the tip."""
        q_kdl = kdl.JntArray(self.n)
        for j in range(self.n):
            q_kdl[self.perm[j]] = q_ur[j]
        jac = kdl.Jacobian(self.n)
        self.jac_solver.JntToJac(q_kdl, jac)
        J = np.empty((6, self.n), dtype=np.float64)
        for r in range(6):
            for c in range(self.n):
                J[r, c] = jac[r, c]
        return J[:, self.perm]   # KDL-order columns -> UR order

    # --------------------------------------------------------------- callbacks
    def js_cb(self, msg):
        self._last_js_names = list(msg.name)
        idx = {name: i for i, name in enumerate(msg.name)}
        try:
            self.q_meas = np.array([msg.position[idx[j]] for j in UR_JOINTS],
                                   dtype=np.float64)
        except (KeyError, IndexError):
            pass  # this /joint_states msg does not carry the UR joints; ignore it

    def twist_cb(self, msg):
        self.latest_twist = np.array([msg.linear.x, msg.linear.y, msg.linear.z,
                                      msg.angular.x, msg.angular.y, msg.angular.z],
                                     dtype=np.float64)
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
        dt = 1.0 / self.rate_hz
        eye6 = np.eye(6)
        lam2 = self.damping ** 2
        max_step = self.max_joint_speed * dt
        while not rospy.is_shutdown():
            if self.q_meas is None:
                rate.sleep()
                continue

            enabled = (not self.require_deadman) or self._button(self.deadman_button)
            fresh = (rospy.Time.now() - self.last_twist_stamp).to_sec() < self.cmd_timeout

            if not enabled:
                # disengage: hold. Drop the target so re-engage re-seeds from the
                # measured pose (no jump). The controller keeps the last command.
                self.q_target = None
                self.vel_smooth[:] = 0.0
                rate.sleep()
                continue

            if self.q_target is None:
                self.q_target = self.q_meas.copy()   # (re)engage from measured pose

            # shaped, smoothed Cartesian velocity command ---------------------
            raw = self.latest_twist if fresh else np.zeros(6)
            vel = self._shape(raw)
            self.vel_smooth = self.vel_alpha * vel + (1.0 - self.vel_alpha) * self.vel_smooth
            if np.linalg.norm(self.vel_smooth) < 1e-4:
                self.vel_smooth[:] = 0.0

            # resolved-rate IK (damped least squares) -------------------------
            #   dq = J^T (J J^T + lambda^2 I)^-1 V
            J = self._jacobian(self.q_meas)
            try:
                dq = J.T.dot(np.linalg.solve(J.dot(J.T) + lam2 * eye6, self.vel_smooth))
            except np.linalg.LinAlgError:
                rospy.logwarn_throttle(1.0, "IK solve failed; holding")
                rate.sleep()
                continue

            # integrate target with a per-joint speed clamp (safety) ----------
            step = np.clip(dq * dt, -max_step, max_step)
            self.q_target = self.q_target + step

            cmd = Float64MultiArray()
            cmd.data = self.q_target.tolist()
            self.cmd_pub.publish(cmd)

            if self.debug:
                rospy.loginfo_throttle(
                    0.5, "enabled=%s fresh=%s V=%s dq=%s",
                    enabled, fresh, np.round(self.vel_smooth, 3).tolist(),
                    np.round(dq, 3).tolist())
            rate.sleep()

    def on_shutdown(self):
        # Do NOT publish a zero target (that would command a jump to all-zeros).
        # Just hand the arm back to a trajectory controller, which holds position.
        if self.manage_controller and self.restore_controller:
            try:
                self.switch_to(self.restore_controller)
            except Exception as e:  # noqa
                rospy.logwarn("failed to restore controller: %s", e)
        rospy.loginfo("spacemouse_servoj_teleop stopped")


if __name__ == "__main__":
    SpaceMouseServoJTeleop().run()
