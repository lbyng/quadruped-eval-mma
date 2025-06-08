#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import signal
import sys
import zmq
import time
import argparse

# Global variable for signal handling
running = True

# Signal handler to ensure clean exit
def signal_handler(sig, frame):
    global running
    print('Shutting down camera...')
    running = False

def main():
    global running
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Single RealSense ZMQ Server')
    parser.add_argument('--fps', type=int, default=15, help='Camera FPS (default: 15)')
    parser.add_argument('--quality', type=int, default=90, 
                       help='JPEG quality 0-100 (default: 90)')
    parser.add_argument('--display', action='store_true', help='Show camera feed in window')
    args = parser.parse_args()
    
    # Initialize ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:9000")
    print(f"Camera ZMQ server started on port 9000")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create RealSense pipeline and config
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Get device list
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) < 1:
            print("Error: No RealSense devices found.")
            return
        
        # Get serial number for the camera
        serial_number = devices[0].get_info(rs.camera_info.serial_number)
        print(f"Found camera with serial number: {serial_number}")
        
        # Configure the camera
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, args.fps)
        
        # Start streaming
        print("Connecting to RealSense camera...")
        pipeline.start(config)
        print("Camera connected successfully!")
        print(f"Streaming at {args.fps} FPS with quality {args.quality}")
        
        # For FPS calculation
        frame_count = 0
        start_time = time.time()
        
        # JPEG encoding parameters
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
        
        # Create display window if requested
        if args.display:
            cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        
        # Main loop
        while running:
            try:
                # Wait for frame
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert image to numpy array
                image = np.asanyarray(color_frame.get_data())
                
                # Encode image
                _, encoded_image = cv2.imencode('.jpg', image, encode_param)
                
                # Send image via ZMQ
                socket.send(encoded_image.tobytes())
                
                # Display camera if requested
                if args.display:
                    cv2.imshow('Camera Feed', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
     
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop streaming
        if 'pipeline' in locals():
            pipeline.stop()
            
        # Clean up display window
        if args.display:
            cv2.destroyAllWindows()
            
        # Clean up ZMQ resources
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()
            
        print("Camera and ZMQ server shutdown complete")

if __name__ == "__main__":
    main()