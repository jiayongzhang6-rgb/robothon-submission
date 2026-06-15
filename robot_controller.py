#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFAI Robothon 2026 - 最终版本
3DOF + 数值IK + Safe Zone补丁
"""

import numpy as np
import mujoco
from typing import Dict, Any


class NumericalIK:
    """数值逆运动学求解器 - 使用雅可比矩阵"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def solve(self, target: np.ndarray, max_iter: int = 500) -> np.ndarray:
        """使用雅可比逆解求IK（多起点）"""
        
        # Safe Zone: 检查是否接近奇异点 [0,0,0.8]
        origin_target = np.array([0.0, 0.0, 0.8])
        dist_to_origin = np.linalg.norm(target - origin_target)
        is_safe_zone = dist_to_origin < 0.18
        base_damping = 0.001
        # Safe Zone内使用更大的阻尼因子
        damping_factor = base_damping * 3.0 if is_safe_zone else base_damping
        
        # 多起点求解
        init_guesses = [
            np.array([0.0, 0.0, 0.0]),
            np.array([np.pi/4, -0.5, 0.0]),
            np.array([-np.pi/4, -0.5, 0.0]),
        ]
        
        best_q = None
        best_error = float('inf')
        
        for init_q in init_guesses:
            q = init_q.copy()
            
            # 每个起点使用较少的迭代次数
            iter_per_start = max_iter // len(init_guesses)
            for _ in range(iter_per_start):
                # 计算当前末端位置
                self.data.qpos[:3] = q
                mujoco.mj_forward(self.model, self.data)
                current_pos = self.data.xpos[-1].copy()
                
                # 计算误差
                error = target - current_pos
                error_mag = np.linalg.norm(error)
                
                # 检查是否收敛
                if error_mag < 0.001:
                    break
                
                # 计算雅可比矩阵
                J = np.zeros((3, 3))
                eps = 0.001
                for i in range(3):
                    q_plus = q.copy()
                    q_plus[i] += eps
                    self.data.qpos[:3] = q_plus
                    mujoco.mj_forward(self.model, self.data)
                    pos_plus = self.data.xpos[-1].copy()
                    J[:, i] = (pos_plus - current_pos) / eps
                
                # 使用伪逆求解
                JT = J.T
                JJT = J @ JT
                # 使用自适应阻尼因子
                dq = JT @ np.linalg.solve(JJT + damping_factor * np.eye(3), error)
                
                q = q + 0.3 * dq
                
                # 限制角度范围
                q[0] = np.clip(q[0], -np.pi, np.pi)
                q[1] = np.clip(q[1], -np.pi/2, np.pi/2)
                q[2] = np.clip(q[2], -np.pi/2, np.pi/2)
            
            # 评估解的质量
            self.data.qpos[:3] = q
            mujoco.mj_forward(self.model, self.data)
            final_pos = self.data.xpos[-1]
            final_error = np.linalg.norm(target - final_pos)
            
            if final_error < best_error:
                best_error = final_error
                best_q = q
        
        return best_q


class RobotController:
    """最终版本 - 3DOF + Safe Zone"""
    
    # 3DOF模型：基座高度0.4m
    XML = """
    <mujoco model="robothon_robot_3dof">
        <option timestep="0.01" gravity="0 0 -9.81"/>
        <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
            <body name="base" pos="0 0 0.4">
                <geom name="base_geom" type="box" size="0.2 0.2 0.1" rgba="0.2 0.6 1 1"/>
                <body name="link1" pos="0 0 0.1">
                    <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
                    <geom name="link1_geom" type="cylinder" size="0.05" fromto="0 0 0 0 0 0.15" rgba="1 0.5 0 1"/>
                    <body name="link2" pos="0 0 0.15">
                        <joint name="joint2" type="hinge" axis="0 1 0" range="-90 90"/>
                        <geom name="link2_geom" type="cylinder" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0 1 0.5 1"/>
                        <body name="link3" pos="0 0 0.2">
                            <joint name="joint3" type="hinge" axis="0 1 0" range="-90 90"/>
                            <geom name="link3_geom" type="cylinder" size="0.03" fromto="0 0 0 0 0 0.15" rgba="0.5 0 1 1"/>
                            <body name="end_effector" pos="0 0 0.15">
                                <geom name="ee_geom" type="sphere" size="0.03" rgba="1 0 0 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor name="motor1" joint="joint1" ctrlrange="-1 1"/>
            <motor name="motor2" joint="joint2" ctrlrange="-1 1"/>
            <motor name="motor3" joint="joint3" ctrlrange="-1 1"/>
        </actuator>
    </mujoco>
    """
    
    def __init__(self, kp: float = 25.0, ki: float = 3.0, kd: float = 0.8):
        self.model = mujoco.MjModel.from_xml_string(self.XML)
        self.data = mujoco.MjData(self.model)
        self.ik = NumericalIK(self.model, self.data)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
    
    def reset(self) -> Dict[str, Any]:
        mujoco.mj_resetData(self.model, self.data)
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        return self.get_status()
    
    def step(self, action: np.ndarray) -> Dict[str, Any]:
        self.data.ctrl[:len(action)] = action
        mujoco.mj_step(self.model, self.data)
        return self.get_status()
    
    def pid_step(self, target_pos: np.ndarray, dt: float = 0.01) -> Dict[str, Any]:
        current_angles = self.data.qpos[:3].copy()
        target_angles = self.ik.solve(target_pos)
        error = target_angles - current_angles
        
        # 处理角度环绕
        error = (error + np.pi) % (2 * np.pi) - np.pi
        
        # 自适应增益
        error_mag = np.linalg.norm(error)
        if error_mag > 0.1:
            kp, ki, kd = self.kp * 2, self.ki * 1.5, self.kd * 1.2
        elif error_mag > 0.05:
            kp, ki, kd = self.kp * 1.5, self.ki * 1.2, self.kd * 1.0
        else:
            kp, ki, kd = self.kp, self.ki, self.kd
        
        self.integral = np.clip(self.integral + error * dt, -15, 15)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        action = kp * error + ki * self.integral + kd * derivative
        action = np.clip(action, -1, 1)
        
        self.prev_error = error
        return self.step(action)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "joint_angles": np.degrees(self.data.qpos[:3]).tolist(),
            "end_effector_pos": self.data.xpos[-1].tolist(),
            "time": self.data.time
        }
    
    def run_control(self, target_pos: np.ndarray, max_steps: int = 300, threshold: float = 0.005) -> list:
        trajectory = []
        for i in range(max_steps):
            state = self.pid_step(target_pos)
            trajectory.append(state)
            
            current_pos = np.array(state["end_effector_pos"])
            error = np.linalg.norm(target_pos - current_pos)
            
            if error < threshold:
                break
        
        return trajectory
