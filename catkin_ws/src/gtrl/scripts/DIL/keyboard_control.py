#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:13:09 2023

@author: oscar
"""

# Copyright (c) 2011, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Willow Garage, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import time
import rospy
import sys, select, termios, tty

from geometry_msgs.msg import Twist

msg = """
Control Your Robot!
---------------------------
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease linear velocity
a/d : increase/decrease angular velocity
space key, s : force stop

CTRL-C to quit
"""
class TeleKey():
    def __init__(self):
        self.twist = Twist()

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
    
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    
    def vels(self, target_linear_vel, target_angular_vel):
        return "currently:\tlinear vel %s\t angular vel %s " % (target_linear_vel,target_angular_vel)

def cmd_callback(cmd):
    global backup_linear_vel, backup_angular_vel
    backup_linear_vel = cmd.linear.x
    backup_angular_vel = cmd.angular.z

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('turtlebot3_teleop')
    pub = rospy.Publisher('/scout/telekey', Twist, queue_size=5)
    cmd = rospy.Subscriber('/scout/cmd_vel', Twist, cmd_callback, queue_size=1)

    status = 0
    target_linear_vel = 0
    target_angular_vel = 0
    backup_linear_vel = 0
    backup_angular_vel = 0
    linear_vel_limit = 1.0
    angular_vel_limit = 1.0
    telekey = TeleKey()
    flag = False
    try:
        print (msg)
        while(1):
            key = telekey.getKey()
            if key == '1' :
                target_linear_vel = backup_linear_vel
                target_angular_vel = backup_angular_vel
                telekey.twist.angular.x = 1
                flag = True
                print('Engage!!!')
            elif key == '2' :
                telekey.twist.angular.x = 0
                flag = False
                print('DisEngage!!!')
            elif key == '\x03' :
                break

            if flag:
                if key == 'w' :
                    if (target_linear_vel + 0.05)*target_linear_vel < 0:
                        target_linear_vel = 0.0
                    else:
                        target_linear_vel = target_linear_vel + 0.05
                    print (telekey.vels(0.5*(target_linear_vel+1),target_angular_vel))
                elif key == 's' :
                    if (target_linear_vel - 0.05)*target_linear_vel < 0:
                        target_linear_vel = 0.0
                    else:
                        target_linear_vel = target_linear_vel - 0.05
                    print (telekey.vels(0.5*(target_linear_vel+1),target_angular_vel))
                elif key == 'a' :
                    if (target_angular_vel + 0.1)*target_angular_vel < 0:
                        target_angular_vel = 0.0
                    else:
                        target_angular_vel = target_angular_vel + 0.1
                    print (telekey.vels(0.5*(target_linear_vel),target_angular_vel))
                elif key == 'd' :
                    if (target_angular_vel - 0.1)*target_angular_vel < 0:
                        target_angular_vel = 0.0
                    else:
                        target_angular_vel = target_angular_vel - 0.1
                    print (telekey.vels(0.5*(target_linear_vel+1),target_angular_vel))
                elif key == 'x' :
                    target_linear_vel   = 0
                    target_angular_vel  = 0
                    print (telekey.vels(0.5, 0))
                elif key == 'q' :
                    target_angular_vel = 0
                    print (telekey.vels(0.5*(target_linear_vel+1), 0))
                elif key == ' ' :
                    target_linear_vel   = -1
                    target_angular_vel  = 0
                    print (telekey.vels(0, 0))

            if target_linear_vel >= 0:
                target_linear_vel = min(target_linear_vel, linear_vel_limit)
            else:
                target_linear_vel = max(target_linear_vel, -linear_vel_limit)
            
            if target_angular_vel >= 0:
                target_angular_vel = min(target_angular_vel, angular_vel_limit)
            else:
                target_angular_vel = max(target_angular_vel, -angular_vel_limit)
            
            telekey.twist.linear.x = target_linear_vel
            telekey.twist.angular.z = target_angular_vel

            pub.publish(telekey.twist)
    except:
        print ('니 내 누긴지 아니? ')

    finally:
        twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        pub.publish(twist)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
