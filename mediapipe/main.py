import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class RealTime3DHandPose:
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 3D plotting setup
        self.fig = plt.figure(figsize=(12, 6))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)
        
        # Initialize 3D plot
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y') 
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Hand Pose')
        
        # Set plot limits
        self.ax_3d.set_xlim([-0.1, 0.1])
        self.ax_3d.set_ylim([-0.1, 0.1])
        self.ax_3d.set_zlim([-0.1, 0.1])
        
        # Initialize 2D image display
        self.ax_2d.set_title('Camera Feed')
        self.ax_2d.axis('off')
        
        # Storage for landmarks
        self.latest_landmarks = None
        self.latest_image = None
        
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
        
        # Store latest results
        self.latest_image = cv2.flip(image_bgr, 1)  # Flip for selfie view
        self.latest_landmarks = results.multi_hand_world_landmarks
        
        return results
    
    def plot_3d_landmarks(self):
        """Plot 3D hand landmarks"""
        self.ax_3d.clear()
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Hand Pose')
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
                
                # Plot landmarks as points
                self.ax_3d.scatter(
                    landmarks_3d[:, 0], 
                    landmarks_3d[:, 1], 
                    landmarks_3d[:, 2],
                    c='red', s=50, alpha=0.8
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
                            'b-', linewidth=2, alpha=0.7
                        )
                
                # Print some landmark coordinates for debugging
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
        
        # Plot 3D landmarks
        self.plot_3d_landmarks()
        
        plt.draw()
        plt.pause(0.001)
    
    def run(self):
        """Main loop for real-time processing"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting real-time 3D hand pose detection...")
        print("Press 'q' on the OpenCV window or close the matplotlib window to quit")
        
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
                if frame_count % 2 == 0:  # Update every 2nd frame
                    self.update_display()
                
                # Also show OpenCV window for immediate feedback
                cv2.imshow('MediaPipe Hands (Press q to quit)', self.latest_image)
                
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
    """Main function to run the 3D hand pose detection"""
    detector = RealTime3DHandPose()
    detector.run()

if __name__ == "__main__":
    main()
