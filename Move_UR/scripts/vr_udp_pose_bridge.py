#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
VR UDP pose bridge.

Receives the JSON packets printed by:

    nc -ul 5005

Only the configured hand role is used. Its localPose is published directly as:

    geometry_msgs/PoseStamped on /wrist_relative_to_waist

The downstream waist_wrist_servoj_teleop.py node captures the first hand pose as
the neutral pose, then maps later offsets from that neutral pose into TCP
velocity and streams servoJ joint targets.
"""

from __future__ import print_function

import json
import socket

import rospy
from geometry_msgs.msg import PoseStamped


class VrUdpPoseBridge(object):
    def __init__(self):
        rospy.init_node("vr_udp_pose_bridge")
        self.host = rospy.get_param("~host", "0.0.0.0")
        self.port = int(rospy.get_param("~port", 5005))
        self.hand_role = rospy.get_param("~hand_role", "RIGHT_HAND")
        self.output_topic = rospy.get_param("~output_topic", "/wrist_relative_to_waist")
        self.frame_id = rospy.get_param("~frame_id", "vr_headset")
        self.debug = rospy.get_param("~debug", False)

        self.pub = rospy.Publisher(self.output_topic, PoseStamped, queue_size=1)
        self.decoder = json.JSONDecoder()
        self.buffer = ""

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.2)

        rospy.loginfo("vr_udp_pose_bridge listening on udp://%s:%d, role=%s -> %s",
                      self.host, self.port, self.hand_role, self.output_topic)

    def _extract_json_objects(self, text):
        self.buffer += text
        objects = []
        while self.buffer:
            start = self.buffer.find("{")
            if start < 0:
                self.buffer = ""
                break
            if start > 0:
                self.buffer = self.buffer[start:]
            try:
                obj, end = self.decoder.raw_decode(self.buffer)
            except ValueError:
                if len(self.buffer) > 1024 * 1024:
                    rospy.logwarn("dropping oversized partial JSON buffer")
                    self.buffer = ""
                break
            objects.append(obj)
            self.buffer = self.buffer[end:].lstrip()
        return objects

    def _publish_hand_pose(self, packet):
        role = packet.get("role", {})
        pose = role.get("localPose", {})
        pos = pose.get("pos")
        rot = pose.get("rotQ")
        if pos is None or rot is None or len(pos) != 3 or len(rot) != 4:
            return

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = -float(pos[0])
        msg.pose.position.y = -float(pos[2])
        msg.pose.position.z = float(pos[1])
        msg.pose.orientation.x = float(rot[0])
        msg.pose.orientation.y = float(rot[1])
        msg.pose.orientation.z = float(rot[2])
        msg.pose.orientation.w = float(rot[3])
        self.pub.publish(msg)

        if self.debug:
            rospy.loginfo_throttle(
                0.5, "frame=%s hand_pos=[%.3f %.3f %.3f] q=[%.3f %.3f %.3f %.3f]",
                packet.get("frame", "?"), float(pos[0]), float(pos[1]), float(pos[2]),
                float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))

    def handle_packet(self, packet):
        if not packet.get("valid", False):
            return
        if packet.get("packetType") != "robotRole":
            return
        role = packet.get("role", {})
        name = role.get("returnedRole") or role.get("requestedRole")
        if name != self.hand_role:
            return
        self._publish_hand_pose(packet)

    def spin(self):
        while not rospy.is_shutdown():
            try:
                data, _addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except socket.error as err:
                rospy.logwarn("UDP recv failed: %s", err)
                continue
            if not data:
                continue
            for obj in self._extract_json_objects(data.decode("utf-8", "ignore")):
                self.handle_packet(obj)


if __name__ == "__main__":
    VrUdpPoseBridge().spin()
