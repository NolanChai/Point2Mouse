import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from collections import deque
import pyautogui
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Disable pyautogui failsafe for smooth operation
pyautogui.FAILSAFE = False

class RealTime3DHandPose:
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # # 3D plotting setup
        # self.fig = plt.figure(figsize=(12, 6))
        # self.ax_3d = self.fig.add_subplot(121, projection='3d')
        # self.ax_2d = self.fig.add_subplot(122)
        
        # # Initialize 3D plot
        # self.ax_3d.set_xlabel('X')
        # self.ax_3d.set_ylabel('Y') 
        # self.ax_3d.set_zlabel('Z')
        # self.ax_3d.set_title('3D Hand Pose')
        
        # # Set plot limits
        # self.ax_3d.set_xlim([-0.1, 0.1])
        # self.ax_3d.set_ylim([-0.1, 0.1])
        # self.ax_3d.set_zlim([-0.1, 0.1])
        
        # # Initialize 2D image display
        # self.ax_2d.set_title('Camera Feed')
        # self.ax_2d.axis('off')
        
        # Storage for landmarks
        self.latest_landmarks = None
        self.latest_image = None
        
        # Mouse control variables
        self.mouse_control_active = False
        self.prev_mouse_pos = None
        self.click_threshold = 0.02  # Threshold for click detection (finger bend)
        self.is_clicking = False
        self.click_cooldown = 0
        self.smoothing_factor = 0.8  # For smoothing cursor movement
        self.sensitivity_multiplier = 1.5  # Increase sensitivity for easier edge access
        
        # Relative movement tracking
        self.initial_cursor_pos = None  # Cursor position when control activates
        self.initial_hand_pos = None    # Hand position when control activates
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Hand connections for 3D plotting
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 17)  # Palm connection
        ]
        
        plt.ion()  # Interactive mode
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt(
            (point1[0] - point2[0])**2 + 
            (point1[1] - point2[1])**2 + 
            (point1[2] - point2[2])**2
        )
    
    def is_fist_with_index_extended(self, landmarks):
        """Detect if hand is making a fist with index finger extended"""
        if len(landmarks) < 21:
            return False
        
        # Key landmark indices
        index_tip = 8
        index_pip = 6
        index_mcp = 5
        middle_tip = 12
        ring_tip = 16
        pinky_tip = 20
        wrist = 0
        
        # Check if index finger is extended (tip is further from wrist than PIP joint)
        index_extended = (
            self.calculate_distance(landmarks[index_tip], landmarks[wrist]) + 0.02 >
            self.calculate_distance(landmarks[index_pip], landmarks[wrist]) 
        )

        # Check if other fingers are folded (tips are closer to wrist than MCP joints)
        middle_folded = (
            self.calculate_distance(landmarks[middle_tip], landmarks[wrist]) <
            self.calculate_distance(landmarks[10], landmarks[wrist])  # Middle MCP
        )
        
        ring_folded = (
            self.calculate_distance(landmarks[ring_tip], landmarks[wrist]) <
            self.calculate_distance(landmarks[14], landmarks[wrist])  # Ring MCP
        )
        
        pinky_folded = (
            self.calculate_distance(landmarks[pinky_tip], landmarks[wrist]) <
            self.calculate_distance(landmarks[18], landmarks[wrist])  # Pinky MCP
        )
        
        # Check if index finger is pointing toward camera (negative Z direction)
        index_pointing_forward = landmarks[index_tip][2] - 0.05 < landmarks[index_mcp][2] 
        
        return index_extended and middle_folded and ring_folded and pinky_folded and index_pointing_forward
    
    def is_index_finger_bent(self, landmarks):
        """Detect if index finger is bent down for clicking"""
        if len(landmarks) < 21:
            return False
        
        index_tip = 8
        index_pip = 6
        index_mcp = 5
        
        # Calculate the bend angle - if tip is significantly lower than PIP joint
        bend_distance = landmarks[index_pip][1] - landmarks[index_tip][1]
        return bend_distance > self.click_threshold
    
    def map_hand_to_screen(self, hand_pos, image_width, image_height):
        """Map hand position to screen coordinates with relative movement"""
        if self.initial_hand_pos is None or self.initial_cursor_pos is None:
            # If we don't have initial positions, use current cursor position
            current_cursor = pyautogui.position()
            self.initial_cursor_pos = (current_cursor.x, current_cursor.y)
            self.initial_hand_pos = (hand_pos[0], hand_pos[1])
            return self.initial_cursor_pos
        
        # Calculate relative movement from initial hand position
        hand_delta_x = (hand_pos[0] - self.initial_hand_pos[0]) * self.sensitivity_multiplier
        hand_delta_y = (hand_pos[1] - self.initial_hand_pos[1]) * self.sensitivity_multiplier
        
        # Apply movement relative to initial cursor position
        screen_x = int(self.initial_cursor_pos[0] - (hand_delta_x * self.screen_width))
        screen_y = int(self.initial_cursor_pos[1] + (hand_delta_y * self.screen_height))
        
        # Clamp to screen boundaries
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return screen_x, screen_y
    
    def control_mouse(self, landmarks, image_width, image_height):
        """Control mouse cursor based on hand landmarks"""
        if not landmarks or len(landmarks) < 21:
            return
        
        # Use index finger MCP joint as reference point for cursor control
        index_mcp = landmarks[5]  # Index finger MCP joint
        
        # Map to screen coordinates
        screen_x, screen_y = self.map_hand_to_screen(index_mcp, image_width, image_height)
        
        # Apply smoothing
        if self.prev_mouse_pos is not None:
            screen_x = int(self.prev_mouse_pos[0] * self.smoothing_factor + screen_x * (1 - self.smoothing_factor))
            screen_y = int(self.prev_mouse_pos[1] * self.smoothing_factor + screen_y * (1 - self.smoothing_factor))
        
        self.prev_mouse_pos = (screen_x, screen_y)
        
        # Move cursor
        try:
            pyautogui.moveTo(screen_x, screen_y)
        except:
            pass  # Ignore any pyautogui errors
        
        # Handle clicking
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        
        finger_bent = self.is_index_finger_bent(landmarks)
        
        if finger_bent and not self.is_clicking and self.click_cooldown == 0:
            try:
                pyautogui.click()
                self.is_clicking = True
                self.click_cooldown = 10  # Prevent rapid clicking
                print("Click!")
            except:
                pass
        elif not finger_bent:
            self.is_clicking = False
    
    def process_frame(self, image):
        """Process a single frame and extract hand landmarks"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Convert back to BGR for display
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Check for mouse control gesture
        if results.multi_hand_landmarks: # multi_hand_world_landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    landmarks_3d.append([landmark.x, landmark.y, landmark.z])
                
                # Check if making control gesture
                if self.is_fist_with_index_extended(landmarks_3d):
                    if not self.mouse_control_active:
                        print("Mouse control activated!")
                    self.mouse_control_active = True
                    self.control_mouse(landmarks_3d, image.shape[1], image.shape[0])
                else:
                    if self.mouse_control_active:
                        print("Mouse control deactivated")
                        # Reset initial positions when control is deactivated
                        self.initial_cursor_pos = None
                        self.initial_hand_pos = None
                    self.mouse_control_active = False
                    self.prev_mouse_pos = None
        else:
            if self.mouse_control_active:
                print("Mouse control deactivated")
                # Reset initial positions when control is deactivated
                self.initial_cursor_pos = None
                self.initial_hand_pos = None
            self.mouse_control_active = False
            self.prev_mouse_pos = None
        
        # Draw 2D landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Flip for selfie view first
        self.latest_image = cv2.flip(image_bgr, 1)
        
        # Add status text AFTER flipping so it appears correctly
        status_text = "MOUSE CONTROL: ON" if self.mouse_control_active else "MOUSE CONTROL: OFF"
        status_color = (0, 255, 0) if self.mouse_control_active else (0, 0, 255)
        cv2.putText(self.latest_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        if self.mouse_control_active:
            cv2.putText(self.latest_image, "Make fist + point index finger at camera", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(self.latest_image, "Bend index finger down to click", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(self.latest_image, "Cursor moves relative to hand position", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Store latest results
        self.latest_landmarks = results.multi_hand_world_landmarks
        
        return results
    
    def plot_3d_landmarks(self):
        """Plot 3D hand landmarks"""
        self.ax_3d.clear()
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        
        title = '3D Hand Pose'
        if self.mouse_control_active:
            title += ' - MOUSE CONTROL ACTIVE'
        self.ax_3d.set_title(title)
        
        self.ax_3d.set_xlim([-0.1, 0.1])
        self.ax_3d.set_ylim([-0.1, 0.1])
        self.ax_3d.set_zlim([-0.1, 0.1])
        
        if self.latest_landmarks:
            for hand_landmarks in self.latest_landmarks:
                # Extract 3D coordinates
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    landmarks_3d.append([landmark.x, landmark.y, landmark.z])
                
                landmarks_3d = np.array(landmarks_3d)
                
                # Color points based on mouse control status
                point_color = 'lime' if self.mouse_control_active else 'red'
                line_color = 'green' if self.mouse_control_active else 'blue'
                
                # Plot landmarks as points
                self.ax_3d.scatter(
                    landmarks_3d[:, 0], 
                    landmarks_3d[:, 1], 
                    landmarks_3d[:, 2],
                    c=point_color, s=50, alpha=0.8
                )
                
                # Highlight index finger tip if mouse control is active
                if self.mouse_control_active:
                    index_tip = landmarks_3d[8]
                    self.ax_3d.scatter(
                        [index_tip[0]], [index_tip[1]], [index_tip[2]],
                        c='yellow', s=100, alpha=1.0
                    )
                
                # Draw connections
                for connection in self.connections:
                    if connection[0] < len(landmarks_3d) and connection[1] < len(landmarks_3d):
                        start_point = landmarks_3d[connection[0]]
                        end_point = landmarks_3d[connection[1]]
                        
                        self.ax_3d.plot3D(
                            [start_point[0], end_point[0]],
                            [start_point[1], end_point[1]], 
                            [start_point[2], end_point[2]],
                            color=line_color, linewidth=2, alpha=0.7
                        )
                
                # Print some landmark coordinates for debugging
                if self.mouse_control_active:
                    index_tip = landmarks_3d[8]  # Index finger tip
                    print(f"Index finger tip 3D: ({index_tip[0]:.3f}, {index_tip[1]:.3f}, {index_tip[2]:.3f})")
    
    def update_display(self):
        """Update the matplotlib display"""
        if self.latest_image is not None:
            # Display camera feed
            self.ax_2d.clear()
            self.ax_2d.set_title('Camera Feed')
            self.ax_2d.axis('off')
            self.ax_2d.imshow(cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB))
        
        # # Plot 3D landmarks
        # self.plot_3d_landmarks()
        
        plt.draw()
        plt.pause(0.001)
    
    def run(self):
        """Main loop for real-time processing"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting real-time 3D hand pose detection with mouse control...")
        print("Make a fist with your index finger pointed at the camera to activate mouse control")
        print("Cursor will move relative to your hand movements from where it was when activated")
        print("Bend your index finger down slightly to click")
        print("Press 'q' on the OpenCV window or close the matplotlib window to quit")
        
        # Create a resizable window
        window_name = 'MediaPipe Hands - Point2Mouse (Press q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, 800, 600)  # Set initial size
        
        try:
            frame_count = 0
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                # Process the frame
                results = self.process_frame(image)
                
                # Update display every few frames to maintain performance
                # if frame_count % 2 == 0:  # Update every 2nd frame
                #     self.update_display()
                
                # Also show OpenCV window for immediate feedback
                cv2.imshow(window_name, self.latest_image)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.close('all')
            self.hands.close()

def main():
    """Main function to run the 3D hand pose detection with mouse control"""
    detector = RealTime3DHandPose()
    detector.run()

if __name__ == "__main__":
    main()
