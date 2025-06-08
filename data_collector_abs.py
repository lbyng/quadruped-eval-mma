import time
import sys
import numpy as np
import zmq
import json
import threading
import cv2
import os
from datetime import datetime
from scipy import interpolate
from tqdm import tqdm

import utils.config as config


class D1DataRecorder:
    def __init__(self):
        self.recording = False
        self.start_time = None
        
        # Output directory for images and JSON
        self.output_dir = config.HDF5_DIR
        self.current_episode_id = 0
        
        # All episodes data
        self.all_episodes_data = []
        
        # Base velocity data from spacemouse - raw
        self.base_velocities_raw = []  # [x, y, yaw]
        self.base_velocities_timestamps_raw = []
        
        # D1 robotic arm command positions - raw
        self.d1_command_positions_raw = []    
        self.d1_command_timestamps_raw = []   
        
        # Wrist camera frame storage - raw
        self.wrist_camera_frames_raw = []           
        self.wrist_camera_timestamps_raw = []       
        
        # Synchronized data storage
        self.sync_timestamps = []
        self.base_velocities = []
        self.d1_command_positions = []
        self.wrist_camera_frames = []
        
        # Recording counters and status
        self.base_velocity_count = 0
        self.d1_command_count = 0
        self.wrist_camera_frame_count = 0
        
        # ZMQ context and sockets
        self.zmq_context = zmq.Context()
        self.zmq_spacemouse_socket = None
        self.zmq_command_socket = None
        self.zmq_wrist_camera_socket = None
        
        # Threads
        self.zmq_spacemouse_thread = None
        self.zmq_command_thread = None
        self.zmq_wrist_camera_thread = None
        self.stop_event = threading.Event()
        
        # Synchronization
        self.sync_sample_rate = config.SYNC_RATE
        self.sync_interval = 1.0 / self.sync_sample_rate
        
        # Camera params
        self.display_camera_frames = config.CAMERA_DISPLAY
        
        # Task description
        self.task_description = ""

    def Init(self):
        # Initialize ZMQ connection for spacemouse data (6000)
        self.zmq_spacemouse_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_spacemouse_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_spacemouse_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_spacemouse_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_spacemouse_socket.connect(config.GO2_CMD_ADDRESS)
        
        # Initialize ZMQ connection for D1 arm commands (5555)
        self.zmq_command_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_command_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_command_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_command_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_command_socket.connect(config.D1_CMD_ADDRESS)
        
        # Initialize ZMQ connection for wrist camera (5558)
        self.zmq_wrist_camera_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_wrist_camera_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_wrist_camera_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.zmq_wrist_camera_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_wrist_camera_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_wrist_camera_socket.connect(config.WRIST_CAMERA_ADDRESS)
        
        print(f"Data will be synchronized to {self.sync_sample_rate}Hz and saved to JSON")        
        
        # Test wrist camera connection
        try:
            self.zmq_wrist_camera_socket.recv(flags=zmq.NOBLOCK)
            print("Wrist camera connected successfully!")
        except zmq.Again:
            print("Waiting for wrist camera data...")

    def SpacemouseThread(self):
        """Thread to receive base velocity from spacemouse."""
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    data = self.zmq_spacemouse_socket.recv_json(flags=zmq.NOBLOCK)
                    
                    # Extract x, y, yaw velocities (already delta values)
                    velocities = [data["x"], data["y"], data["yaw"]]
                    
                    # Record velocities and timestamp
                    self.base_velocities_raw.append(velocities)
                    self.base_velocities_timestamps_raw.append(time.time() - self.start_time)
                    self.base_velocity_count += 1
                    
                except zmq.Again:
                    pass
                except Exception as e:
                    print(f"Spacemouse thread error: {e}")

                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"Spacemouse thread error: {str(e)}")
        
        print("Spacemouse thread ended")

    def D1CommandThread(self):
        """Thread to receive and record D1 command positions."""
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    msg = self.zmq_command_socket.recv_string(flags=zmq.NOBLOCK)
                    data = json.loads(msg)
                    
                    if "positions" in data:
                        # Record joint positions and timestamp
                        self.d1_command_positions_raw.append(data["positions"])
                        self.d1_command_timestamps_raw.append(time.time() - self.start_time)
                        self.d1_command_count += 1
                except zmq.Again:
                    pass
                except json.JSONDecodeError:
                    pass

                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"D1 command thread error: {str(e)}")
        
        print("D1 command thread ended")

    def WristCameraThread(self):
        """Thread to receive and record wrist camera frames."""
        # For FPS calculation
        fps_frame_count = 0
        fps_start_time = time.time()
        fps = 0
        display_window_created = False
        
        # Main thread loop
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    jpg_buffer = self.zmq_wrist_camera_socket.recv(flags=zmq.NOBLOCK)
                    
                    img_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        current_time = time.time()
                        fps_frame_count += 1
                        fps_elapsed = current_time - fps_start_time
                        
                        if fps_elapsed >= 1.0:
                            fps = fps_frame_count / fps_elapsed
                            fps_frame_count = 0
                            fps_start_time = current_time
                        
                        # Save frame
                        self.wrist_camera_frames_raw.append(frame.copy())
                        self.wrist_camera_timestamps_raw.append(current_time - self.start_time)
                        self.wrist_camera_frame_count += 1
                        
                        # Display frame if enabled
                        if self.display_camera_frames:
                            display_frame = frame.copy()
                            cv2.putText(
                                display_frame, 
                                f"Wrist Camera - Frame: {self.wrist_camera_frame_count} FPS: {fps:.1f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                            )
                            
                            if not display_window_created:
                                cv2.namedWindow('Wrist Camera', cv2.WINDOW_NORMAL)
                                display_window_created = True
                                
                            cv2.imshow('Wrist Camera', display_frame)
                            cv2.waitKey(1)
                    
                except zmq.Again:
                    pass
                
                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"Wrist camera thread error: {str(e)}")
        
        print("Wrist camera thread ended")

    def UpdateStatus(self):
        """Update recording status"""
        if self.recording:
            elapsed = time.time() - self.start_time
            base_rate = self.base_velocity_count / elapsed if elapsed > 0 else 0
            d1_cmd_rate = self.d1_command_count / elapsed if elapsed > 0 else 0
            wrist_cam_rate = self.wrist_camera_frame_count / elapsed if elapsed > 0 else 0
            
            status = f"\rRec: {elapsed:.1f}s | Base vel: {self.base_velocity_count} ({base_rate:.1f}Hz) | "
            status += f"D1 cmd: {self.d1_command_count} ({d1_cmd_rate:.1f}Hz) | "
            status += f"Wrist: {self.wrist_camera_frame_count} ({wrist_cam_rate:.1f}fps)"
            
            print(status, end='', flush=True)

    def StartRecording(self):
        if self.recording:
            print("Already recording...")
            return
        
        # Get task description from config
        self.task_description = config.TASK
        
        # Clear previous data
        self._clear_all_data()
        
        # Start recording
        self.start_time = time.time()
        self.recording = True
        
        # Reset stop event and start threads
        self.stop_event.clear()
        
        # Start thread for spacemouse velocities
        self.zmq_spacemouse_thread = threading.Thread(target=self.SpacemouseThread)
        self.zmq_spacemouse_thread.daemon = True
        self.zmq_spacemouse_thread.start()
        
        # Start thread for D1 command positions
        self.zmq_command_thread = threading.Thread(target=self.D1CommandThread)
        self.zmq_command_thread.daemon = True
        self.zmq_command_thread.start()
        
        # Start thread for wrist camera
        self.zmq_wrist_camera_thread = threading.Thread(target=self.WristCameraThread)
        self.zmq_wrist_camera_thread.daemon = True
        self.zmq_wrist_camera_thread.start()
        
        print(f"Starting data recording for task: '{self.task_description}'...")

    def _clear_all_data(self):
        """Clear all data arrays"""
        # Clear raw data
        self.base_velocities_raw = []
        self.base_velocities_timestamps_raw = []
        
        self.d1_command_positions_raw = []
        self.d1_command_timestamps_raw = []
        
        self.wrist_camera_frames_raw = []
        self.wrist_camera_timestamps_raw = []
        
        # Clear synchronized data
        self.sync_timestamps = []
        self.base_velocities = []
        self.d1_command_positions = []
        self.wrist_camera_frames = []
        
        # Reset counters
        self.base_velocity_count = 0
        self.d1_command_count = 0
        self.wrist_camera_frame_count = 0

    def StopRecording(self, save_data=True):
        if not self.recording:
            print("No active recording...")
            return
        
        # Stop recording
        self.recording = False
        self.stop_event.set()
        
        # Wait for threads to finish
        threads = [
            (self.zmq_spacemouse_thread, "Spacemouse"),
            (self.zmq_command_thread, "D1 command"),
            (self.zmq_wrist_camera_thread, "Wrist camera")
        ]
        
        for thread, name in threads:
            if thread and thread.is_alive():
                print(f"Waiting for {name} thread to finish...")
                thread.join(timeout=2.0)
        
        # Calculate total elapsed time from raw data
        total_elapsed = self._calculate_total_elapsed_time()
        
        # Print recording statistics
        self._print_recording_statistics(total_elapsed)
        
        if not save_data:
            print("Data discarded as requested.")
            return
        
        # Synchronize data to common timeline
        print(f"Synchronizing all data to {self.sync_sample_rate}Hz timeline...")
        self.SynchronizeData()
        
        # Save data if any was recorded
        has_data = (self.base_velocity_count > 0 or self.d1_command_count > 0 or 
                   self.wrist_camera_frame_count > 0)
        
        if has_data:
            json_file = self.SaveToJSON()
            if json_file:
                print(f"Data successfully saved to: {json_file}")
                self.current_episode_id += 1
            else:
                print("Failed to save JSON file")
        else:
            print("No data recorded, not saving file.")

    def _calculate_total_elapsed_time(self):
        """Calculate total recording duration from all data sources"""
        total_elapsed = 0
        if len(self.base_velocities_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.base_velocities_timestamps_raw[-1])
        if len(self.d1_command_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.d1_command_timestamps_raw[-1])
        if len(self.wrist_camera_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.wrist_camera_timestamps_raw[-1])
        return total_elapsed

    def _print_recording_statistics(self, total_elapsed):
        """Print recording statistics"""
        # Calculate actual sampling rates from raw data
        base_sample_rate = self.base_velocity_count / total_elapsed if total_elapsed > 0 else 0
        d1_cmd_sample_rate = self.d1_command_count / total_elapsed if total_elapsed > 0 else 0
        wrist_camera_rate = self.wrist_camera_frame_count / total_elapsed if total_elapsed > 0 else 0
        
        print("")
        print(f"Recording stopped. Total time: {total_elapsed:.2f} seconds")
        print(f"Raw data collection rates:")
        print(f"- Base velocity: {self.base_velocity_count} samples ({base_sample_rate:.1f}Hz)")
        print(f"- D1 cmd: {self.d1_command_count} samples ({d1_cmd_sample_rate:.1f}Hz)")
        print(f"- Wrist camera: {self.wrist_camera_frame_count} frames ({wrist_camera_rate:.1f}fps)")

    def SynchronizeData(self):
        """Synchronize all data to a common timeline using interpolation"""
        # Check if we have any data to synchronize
        if not (len(self.base_velocities_timestamps_raw) > 0 or len(self.d1_command_timestamps_raw) > 0 or 
                len(self.wrist_camera_timestamps_raw) > 0):
            print("No data to synchronize")
            return
            
        # Find the earliest and latest timestamps across all data sources
        start_times = []
        end_times = []
        
        if len(self.base_velocities_timestamps_raw) > 0:
            start_times.append(self.base_velocities_timestamps_raw[0])
            end_times.append(self.base_velocities_timestamps_raw[-1])
            
        if len(self.d1_command_timestamps_raw) > 0:
            start_times.append(self.d1_command_timestamps_raw[0])
            end_times.append(self.d1_command_timestamps_raw[-1])
            
        if len(self.wrist_camera_timestamps_raw) > 0:
            start_times.append(self.wrist_camera_timestamps_raw[0])
            end_times.append(self.wrist_camera_timestamps_raw[-1])
        
        # Use the latest start time and earliest end time to ensure all data is available
        sync_start = max(start_times) if start_times else 0
        sync_end = min(end_times) if end_times else 0
        
        if sync_end <= sync_start:
            print("WARNING: Invalid time range for synchronization")
            return
        
        # Create common timeline at sync Hz
        self.sync_timestamps = np.arange(sync_start, sync_end, self.sync_interval)
        
        # Synchronize base velocity data if available (already delta values, no conversion needed)
        if len(self.base_velocities_timestamps_raw) > 1:
            print("Synchronizing base velocity data...")
            base_timestamps = np.array(self.base_velocities_timestamps_raw)
            base_velocities_array = np.array(self.base_velocities_raw)
            self.base_velocities = self._interpolate_array(base_timestamps, base_velocities_array, self.sync_timestamps)
        
        # Synchronize D1 command data (keep absolute positions)
        if len(self.d1_command_timestamps_raw) > 1 and len(self.d1_command_positions_raw) > 0:
            print("Synchronizing D1 command data...")
            d1_cmd_timestamps = np.array(self.d1_command_timestamps_raw)
            d1_cmd_positions = np.array(self.d1_command_positions_raw)
            
            # Interpolate to get synchronized positions
            self.d1_command_positions = self._interpolate_array(d1_cmd_timestamps, d1_cmd_positions, self.sync_timestamps)
        
        # Synchronize camera frames if available
        if len(self.wrist_camera_timestamps_raw) > 1 and len(self.wrist_camera_frames_raw) > 0:
            print("Synchronizing wrist camera frames...")
            self.wrist_camera_frames = self._nearest_frames(self.wrist_camera_timestamps_raw, self.wrist_camera_frames_raw, self.sync_timestamps)
        
        print(f"Synchronized {len(self.sync_timestamps)} samples at {self.sync_sample_rate}Hz")

    def _interpolate_array(self, src_timestamps, src_values, target_timestamps):
        """Interpolate array data to target timestamps"""
        if len(src_timestamps) != len(src_values):
            print(f"ERROR: Timestamp and value arrays have different lengths: {len(src_timestamps)} vs {len(src_values)}")
            return []
            
        result = []
        # If src_values is 1D, use simple interpolation
        if len(src_values.shape) == 1:
            interp_func = interpolate.interp1d(src_timestamps, src_values, axis=0, 
                                              bounds_error=False, fill_value="extrapolate")
            result = interp_func(target_timestamps)
        # If src_values is 2D, interpolate each component
        elif len(src_values.shape) == 2:
            try:
                interp_func = interpolate.interp1d(src_timestamps, src_values, axis=0, 
                                                bounds_error=False, fill_value="extrapolate")
                result = interp_func(target_timestamps)
            except Exception as e:
                print(f"Interpolation error: {e}")
                # Fallback method: use nearest neighbor for each point
                result = np.zeros((len(target_timestamps), src_values.shape[1]))
                for i, t in enumerate(target_timestamps):
                    idx = np.abs(src_timestamps - t).argmin()
                    result[i] = src_values[idx]
        else:
            print(f"ERROR: Unsupported array shape for interpolation: {src_values.shape}")
            
        return result

    def _nearest_frames(self, src_timestamps, src_frames, target_timestamps):
        """Select nearest camera frames for target timestamps"""
        result = []
        
        # Check if we have enough source frames
        if len(src_timestamps) < 1 or len(src_frames) < 1:
            return result
            
        # Convert source timestamps to numpy array if not already
        src_timestamps_array = np.array(src_timestamps)
        
        for target_time in target_timestamps:
            # Find index of nearest timestamp
            idx = np.abs(src_timestamps_array - target_time).argmin()
            # Add the corresponding frame to the result
            result.append(src_frames[idx].copy())
            
        return result

    def SaveToJSON(self):
        """Save synchronized data to JSON format with images"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create episode directory directly in output_dir
        episode_dir = os.path.join(self.output_dir, f"{self.current_episode_id:06d}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Check if we have synchronized data
        if len(self.sync_timestamps) == 0:
            print("No synchronized data to save")
            return None
        
        num_samples = len(self.sync_timestamps)
        print(f"Saving {num_samples} synchronized samples for episode {self.current_episode_id}")
        
        # Save images and create JSON entries for this episode
        for step_id in tqdm(range(num_samples), desc="Processing frames"):
            # Prepare action array
            action = []
            
            # Add D1 absolute positions (7 values)
            if step_id < len(self.d1_command_positions) and len(self.d1_command_positions[step_id]) > 0:
                d1_positions = self.d1_command_positions[step_id][:7]
                action.extend(d1_positions)
            else:
                # Default zeros for D1 if no data
                action.extend([0.0] * 7)
            
            # Add base velocities (3 values)
            if step_id < len(self.base_velocities) and len(self.base_velocities[step_id]) > 0:
                action.extend(self.base_velocities[step_id])
            else:
                # Default zeros for base if no data
                action.extend([0.0] * 3)
            
            # Save wrist camera image
            image_filename = f"{step_id:04d}.png"
            image_path = os.path.join(episode_dir, image_filename)
            
            if step_id < len(self.wrist_camera_frames):
                cv2.imwrite(image_path, self.wrist_camera_frames[step_id])
            
            # Create JSON entry - include full path
            entry = {
                "image": os.path.join(config.HDF5_DIR, f"{self.current_episode_id:06d}", image_filename),
                "task": self.task_description,
                "raw_action": str(action)
            }
            
            self.all_episodes_data.append(entry)
        
        # Save the complete JSON file with all episodes
        json_filename = os.path.join(self.output_dir, "all_episodes.json")
        with open(json_filename, 'w') as f:
            json.dump(self.all_episodes_data, f, indent=2)
        
        print(f"Episode {self.current_episode_id} saved:")
        print(f"  - Images: {episode_dir}")
        print(f"  - Total entries in JSON: {len(self.all_episodes_data)}")
        print(f"  - JSON file: {json_filename}")
        print(f"  - Action format: [d1_joint1_pos, ..., d1_joint7_pos, base_x_vel, base_y_vel, base_yaw_vel]")
        
        return json_filename
    
    def Cleanup(self):
        if self.display_camera_frames:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(0.1)
        
        # Close ZMQ sockets
        sockets = [
            (self.zmq_spacemouse_socket, "Spacemouse"),
            (self.zmq_command_socket, "D1 command"),
            (self.zmq_wrist_camera_socket, "Wrist camera")
        ]
        
        for socket, name in sockets:
            if socket:
                socket.close()
                print(f"Closed {name} socket")
        
        # Terminate ZMQ context
        if self.zmq_context:
            self.zmq_context.term()
            print("ZMQ context terminated")


def main():
    print(f"Data will be synchronized to {config.SYNC_RATE}Hz and saved to JSON")
    print(f"Task: '{config.TASK}'")

    # Create and initialize recorder
    recorder = D1DataRecorder()
    recorder.Init()

    print("\nReady, press Enter to start a new recording...")

    try:
        while True:
            # Wait for user input
            user_input = input().strip().lower()
            
            if not recorder.recording:
                # Start recording regardless of input when not recording
                if user_input == 'quit':
                    # Save final JSON file before exiting
                    if len(recorder.all_episodes_data) > 0:
                        final_json = os.path.join(recorder.output_dir, "all_episodes.json")
                        with open(final_json, 'w') as f:
                            json.dump(recorder.all_episodes_data, f, indent=2)
                        print(f"\nFinal JSON file saved: {final_json}")
                        print(f"Total entries: {len(recorder.all_episodes_data)}")
                    break
                    
                recorder.StartRecording()
                print("Recording... Options:")
                print("  - Press Enter to SAVE")
                print("  - Type 'discard' or 'd' to DISCARD")
            else:
                # Stop recording with different options
                if user_input in ['discard', 'd']:
                    print("Discarding recorded data...")
                    recorder.StopRecording(save_data=False)
                    print("\nReady, press Enter to start a new recording...")
                else:
                    # Default behavior: save data (Enter or any other input)
                    print("Saving recorded data...")
                    recorder.StopRecording(save_data=True)
                    print("\nReady, press Enter to start a new recording...")
                
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
        if recorder.recording:
            print("Discarding current recording due to forced exit...")
            recorder.StopRecording(save_data=False)
        
        # Save final JSON file before exiting
        if len(recorder.all_episodes_data) > 0:
            final_json = os.path.join(recorder.output_dir, "all_episodes.json")
            with open(final_json, 'w') as f:
                json.dump(recorder.all_episodes_data, f, indent=2)
            print(f"\nFinal JSON file saved: {final_json}")
            print(f"Total entries: {len(recorder.all_episodes_data)}")
            
        recorder.Cleanup()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Program terminated by user")
        
    sys.exit(0)


if __name__ == "__main__":
    main()