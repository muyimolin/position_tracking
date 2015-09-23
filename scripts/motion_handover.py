#!/usr/bin/env python

import sys
lib_path = "/home/motion/iml-internal/Ebolabot/"
# model_path = "/home/motion/iml-internal/Ebolabot/klampt_models"
sys.path.append(lib_path)
# sys.path.append(model_path)


from Motion import motion
from Motion import config, config_physical
from klampt import *
from klampt.glprogram import *
import math
import time
import os


class MyGLViewer(GLRealtimeProgram):
    def __init__(self,world):
        GLRealtimeProgram.__init__(self,"My GL program")
        self.world = world
        self.t0 = time.time()
        self.velocity_scale = 2.0
        self.t_current = time.time()
        self.t_previous = time.time()
        self.inPosition = False
        self.ee_current = None
        self.ee_previous = None

    def display(self):
        # Put your display handler here
        # the current example draws the sensed robot in grey and the
        # commanded configurations in transparent green
        robot = motion.robot
        q = robot.getKlamptSensedPosition()
        self.world.robot(0).setConfig(q)
        self.world.drawGL()

        # draw commanded configuration
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,[0,1,0,0.5])
        q = robot.getKlamptCommandedPosition()
        self.world.robot(0).setConfig(q)
        self.world.robot(0).drawGL(False)
        glDisable(GL_BLEND)

    def control_loop(self):
        robot = motion.robot
        # Put your control handler here
        # currently moves a joint according to sinusoid
        # q = robot.left_limb.commandedPosition()
        # print "Left commanded",q
        # q[1] = math.sin(time.time() )
        # robot.left_limb.positionCommand(q)
        t = time.time()-self.t0

        if robot.left_mq.moving():
            pass
        elif not self.inPosition:
            q = robot.left_limb.commandedPosition()
            q[1] = -1.0
            q[3] = 1.0
            q[5] = 1.0
            print "Sent setRamp command"
            robot.left_mq.setRamp(q)
            self.inPosition = True
            print "self.inPosition = ", self.inPosition
            self.t_previous = time.time()
        else:
            # control robot motion - task space control - look into motion lib
            # robot.left_ee.velocityCommand([0,0,0],[math.sin(t)*0.2,math.cos(t)*0.2,0])
            print "End Effector Command: Velocity"
            ang_vel = [0.0, 0.0, 0.0]
            pos_vel = [0.0, 0.0, -0.05]

            # velocityCommand sends an instantaneous velocity.
            # The detected velocity is twice as much as the commanded velocity
            robot.left_ee.velocityCommand(ang_vel, pos_vel)

            # driveCommand sends velocity command that consistently drives
            # the end-effect to follow a screw motion in Cartesian space starting from its current commanded transform.
            # The detected velocity is the commanded velocity

            # robot.left_ee.driveCommand(ang_vel, pos_vel)
            # robot.left_ee.driveCommand([0.0, 0.0, 0.0], [1, 0, 0])
            self.t_current = time.time()
            left_ee_tf = robot.left_ee.commandedTransform()
            self.ee_current = left_ee_tf[1]

            ee_diff = [0.0, 0.0, 0.0]
            duration = (self.t_current - self.t_previous)
            if self.ee_current is not None and self.ee_previous is not None:
                for i in range(3):
                    ee_diff[i] = (self.ee_current[i] - self.ee_previous[i])/duration
                print "velocity", ee_diff
            self.ee_previous = self.ee_current
            self.t_previous = self.t_current
        return

    def idle(self):
        self.control_loop()
        glutPostRedisplay()

    def mousefunc(self,button,state,x,y):
        # Put your mouse handler here
        print "mouse",button,state,x,y
        GLRealtimeProgram.mousefunc(self,button,state,x,y)

    def motionfunc(self,x,y):
        # Put your mouse motion handler here
        GLRealtimeProgram.motionfunc(self,x,y)

    def specialfunc(self,c,x,y):
        # Put your keyboard special character handler here
        print c,"pressed"

    def keyboardfunc(self,c,x,y):
        # Put your keyboard handler here
        # the current example toggles simulation / movie mode
        print c,"pressed"
        if c == 'q':
            motion.robot.shutdown()
            exit(0)
        glutPostRedisplay()


if __name__ == "__main__":
    config_physical.parse_args()
    print "motion_template.py: sets up a visualizer for testing Motion"
    print "control loops.  Press q to exit."
    print

    print "Loading APC Motion model", config_physical.klampt_model
    print "config.mode =", config_physical.mode
    robot = motion.setup(mode=config_physical.mode, klampt_model=lib_path+config_physical.klampt_model, libpath=lib_path)

    res = robot.startup()
    if not res:
        print "Error starting up APC Motion"
        exit(1)
    time.sleep(0.1)
    world = WorldModel()
    res = world.readFile(lib_path+config.klampt_model)
    if not res:
        raise RuntimeError("Unable to load APC Motion model ")
    
    viewer = MyGLViewer(world)
    viewer.run()
