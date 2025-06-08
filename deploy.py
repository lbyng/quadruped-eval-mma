import os
import time
import numpy as np
import requests
import json_numpy
import zmq
import cv2
import threading
json_numpy.patch()

import utils.deploy_config as dconf

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


class CameraReceiver:
    """ZMQ camera stream receiver"""
    def __init__(self):
        self.address = dconf.CAMERA_ADDRESS
        self.zmq_context = zmq.Context()
        self.zmq_socket = None
        self.latest_frame = None
        self.frame_count = 0
        self.receiving = False
        self.thread = None
        self.stop_event = threading.Event()
        
    def start(self):
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.zmq_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_socket.connect(self.address)
        
        print(f"[INFO] Connecting to camera at {self.address}")
        
        try:
            self.zmq_socket.recv(flags=zmq.NOBLOCK)
            print("[INFO] Camera connected successfully!")
        except zmq.Again:
            print("[INFO] Waiting for camera data...")
        
        self.receiving = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _receive_loop(self):
        while self.receiving and not self.stop_event.is_set():
            try:
                jpg_buffer = self.zmq_socket.recv(flags=zmq.NOBLOCK)
                img_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
                # frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
                if frame is not None:
                    self.latest_frame = frame
                    self.frame_count += 1
                    
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[ERROR] Camera receive error: {e}")
                
            time.sleep(0.001)
    
    def get_frame(self):
        return self.latest_frame
    
    def stop(self):
        self.receiving = False
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.zmq_socket:
            self.zmq_socket.close()
            
        if self.zmq_context:
            self.zmq_context.term()
            
        print(f"[INFO] Camera receiver stopped. Total frames: {self.frame_count}")


class RobotController:
    """Robot controller for arm and base"""
    def __init__(self, network_interface="enp14s0", arm_port=5555):
        self.network_interface = network_interface
        self.arm_port = arm_port
        self.sport_client = None
        self.msc = None
        self.initialized = False
        
        # Arm control
        self.zmq_context = zmq.Context()
        self.arm_socket = None
        # self.initial_positions = [-7.35, -20.40, 14.10, 65.90, -16.81, 0.0, 65.0]
        # self.initial_positions = [-2.9625689344938735, 16.100665716819492, 40.34181240641276, -39.00653898982952, -9.098367385443595, 0.0, 43.77417310214136]
        self.initial_positions = [-7.35, -20.40, 14.10, 65.90, -16.81, 0.0, 65.0]
        self.current_positions = self.initial_positions.copy()
        
    def init(self):
        try:
            # Initialize base control
            ChannelFactoryInitialize(0, self.network_interface)
            
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()
            print("[INFO] Robot sport client connected")
            
            self.msc = MotionSwitcherClient()
            self.msc.SelectMode("ai")
            print("[INFO] AI motion mode enabled")
            
            self.sport_client.BalanceStand()
            time.sleep(1.0)
            
            # Initialize arm control
            self.arm_socket = self.zmq_context.socket(zmq.PUB)
            self.arm_socket.bind(f"tcp://*:{self.arm_port}")
            print(f"[INFO] Arm controller publishing on port {self.arm_port}")
            time.sleep(0.5)  # Allow time for binding
            
            # Send initial position
            self.arm_socket.send_json({"positions": self.current_positions})
            
            self.initialized = True
            print("[INFO] Robot controller initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize robot: {e}")
            self.initialized = False
            
    def execute_action(self, action):
        """
        Execute 10-dimensional action
        action[0:6] - arm joint deltas
        action[6] - gripper state (0=open, 1=close)
        action[7:10] - base velocities [x, y, yaw]
        """
        if not self.initialized:
            print("[WARN] Robot not initialized, skipping action")
            return
            
        try:
            # Apply arm deltas for first 6 joints
            arm_deltas = action[:6]
            for i in range(6):
                self.current_positions[i] += float(arm_deltas[i])
            
            # Handle gripper state (joint 7)
            gripper_state = float(action[6])
            if gripper_state >= 0.5:  # Close
                self.current_positions[6] = -30.0
            else:  # Open
                self.current_positions[6] = 65.0
            
            # Send arm positions
            self.arm_socket.send_json({"positions": self.current_positions})
            
            # Execute base velocities
            base_x_vel = float(action[7])
            base_y_vel = float(action[8])
            base_yaw_vel = float(action[9])
            
            # Apply deadzone
            deadzone = getattr(dconf, 'BASE_DEADZONE', 0.05)
            if abs(base_x_vel) < deadzone: base_x_vel = 0
            if abs(base_y_vel) < deadzone: base_y_vel = 0
            if abs(base_yaw_vel) < deadzone: base_yaw_vel = 0
            
            # Clip velocities
            x_vel_range = getattr(dconf, 'BASE_X_VEL_RANGE', [-0.6, 0.6])
            y_vel_range = getattr(dconf, 'BASE_Y_VEL_RANGE', [-0.4, 0.4])
            yaw_vel_range = getattr(dconf, 'BASE_YAW_VEL_RANGE', [-0.8, 0.8])
            
            base_x_vel = np.clip(base_x_vel, x_vel_range[0], x_vel_range[1])
            base_y_vel = np.clip(base_y_vel, y_vel_range[0], y_vel_range[1])
            base_yaw_vel = np.clip(base_yaw_vel, yaw_vel_range[0], yaw_vel_range[1])
            
            # Send base command
            self.sport_client.Move(base_y_vel, base_x_vel, base_yaw_vel)
            
        except Exception as e:
            print(f"[ERROR] Failed to execute action: {e}")
            
    def reset_arm_position(self):
        """Reset arm to initial position"""
        self.current_positions = self.initial_positions.copy()
        if self.arm_socket:
            self.arm_socket.send_json({"positions": self.current_positions})
            
    def emergency_stop(self):
        if self.sport_client:
            try:
                self.sport_client.Move(0, 0, 0)
                self.sport_client.BalanceStand()
                print("[INFO] Emergency stop executed")
            except:
                pass
                
    def cleanup(self):
        if self.sport_client:
            try:
                self.sport_client.Move(0, 0, 0)
                self.sport_client.BalanceStand()
            except:
                pass
                
        if self.arm_socket:
            self.arm_socket.close()
            
        if self.zmq_context:
            self.zmq_context.term()
            
        print("[INFO] Robot controller cleaned up")


def send_request(image_array: np.ndarray, instruction: str, server_url: str):
    """Send image and instruction to inference server"""
    payload = {
        "image": image_array,
        "instruction": instruction
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(server_url, headers=headers, data=json_numpy.dumps(payload))
    
    if response.status_code != 200:
        raise Exception(f"Server error: {response.text}")
    
    print(response.json())
    action = response.json()
    print(type(action))
    action = action.tolist()
    
    if not isinstance(action, list) or len(action) != 10:
        print(type(action))
        raise ValueError(f"Expected 10-dimensional action, got: {action}")
    
    return action


def run_closed_loop_control():
    """Main control loop"""
    # Load configuration
    camera_address = dconf.CAMERA_ADDRESS
    network_interface = dconf.NETWORK_INTERFACE
    arm_port = dconf.ARM_PORT
    server_url = dconf.SERVER_URL
    task_instruction = dconf.TASK_INSTRUCTION
    traj_length = dconf.TRAJ_LENGTH
    step_delay = dconf.STEP_DELAY
    display_camera = dconf.DISPLAY_CAMERA
    
    print("[INFO] Configuration loaded:")
    print(f"  - Camera: {camera_address}")
    print(f"  - Network interface: {network_interface}")
    print(f"  - Arm port: {arm_port}")
    print(f"  - Server: {server_url}")
    print(f"  - Task: {task_instruction}")
    print(f"  - Steps: {traj_length}")
    
    # Initialize camera
    camera_receiver = CameraReceiver()
    camera_receiver.start()
    
    # Initialize robot
    robot_controller = RobotController(network_interface, arm_port)
    robot_controller.init()
    
    if not robot_controller.initialized:
        print("[ERROR] Failed to initialize robot controller")
        camera_receiver.stop()
        return
    
    # Wait for first frame
    print("[INFO] Waiting for first camera frame...")
    wait_start = time.time()
    camera_timeout = getattr(dconf, 'CAMERA_TIMEOUT', 10)
    
    while camera_receiver.get_frame() is None:
        if time.time() - wait_start > camera_timeout:
            print(f"[ERROR] Camera timeout ({camera_timeout}s)")
            camera_receiver.stop()
            robot_controller.cleanup()
            return
        time.sleep(0.1)
    print("[INFO] Camera ready!")
    
    if display_camera:
        cv2.namedWindow('Robot Camera', cv2.WINDOW_NORMAL)
    
    # Main control loop
    print(f"\n[INFO] Starting closed-loop control...")
    print("="*60)
    
    try:
        for step in range(traj_length):
            loop_start = time.time()
            
            current_image = camera_receiver.get_frame()
            
            if current_image is None:
                print(f"[WARN] No camera frame at step {step}")
                continue
            
            if display_camera:
                display_frame = current_image.copy()
                cv2.putText(
                    display_frame, 
                    f"Step: {step+1}/{traj_length}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                cv2.imshow('Robot Camera', display_frame)
                cv2.waitKey(1)
            
            print(f"\nStep {step + 1}/{traj_length}")
            
            try:
                action = send_request(current_image, task_instruction, server_url)
                print(f"Action: {np.array(action).round(3)}")
                print(f"  - Arm deltas: {np.array(action[:6]).round(3)}")
                print(f"  - Gripper: {'close' if action[6] >= 0.5 else 'open'} ({action[6]:.1f})")
                print(f"  - Base vels: {np.array(action[7:10]).round(3)}")
                print(f"  - Arm positions: {np.array(robot_controller.current_positions).round(1)}")
                
                robot_controller.execute_action(action)
                
            except Exception as e:
                print(f"[ERROR] Failed to execute: {e}")
                robot_controller.sport_client.Move(0, 0, 0)
                continue
            
            step_time = time.time() - loop_start
            print(f"Step time: {step_time:.3f}s")
            
            if step_delay > 0:
                time.sleep(step_delay)
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        robot_controller.emergency_stop()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        robot_controller.emergency_stop()
    finally:
        print("\n[INFO] Cleaning up...")
        robot_controller.cleanup()
        camera_receiver.stop()
        if display_camera:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    
    print("\n" + "="*60)
    print("[INFO] Control finished")
    print(f"[INFO] Total frames: {camera_receiver.frame_count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Robot Closed-Loop Control')
    parser.add_argument('--server-url', type=str, help='Override server URL')
    parser.add_argument('--task', type=str, help='Override task instruction')
    parser.add_argument('--steps', type=int, help='Override trajectory length')
    parser.add_argument('--display', action='store_true', help='Enable camera display')
    parser.add_argument('--reset-arm', action='store_true', help='Reset arm to initial position and exit')
    
    args = parser.parse_args()
    
    if args.reset_arm:
        robot = RobotController()
        robot.init()
        robot.reset_arm_position()
        print("[INFO] Arm reset to initial position")
        robot.cleanup()
        return
    
    if args.server_url:
        dconf.SERVER_URL = args.server_url
    if args.task:
        dconf.TASK_INSTRUCTION = args.task
    if args.steps:
        dconf.TRAJ_LENGTH = args.steps
    if args.display:
        dconf.DISPLAY_CAMERA = True
    
    run_closed_loop_control()


if __name__ == "__main__":
    main()