🖐️ Gesture-Flappy: 
Hand-Tracked Bird GameAn interactive remake of the classic Flappy Bird game using Computer Vision. 
Instead of clicking a mouse or tapping a screen, you control the bird's vertical movement by moving your index finger in front of your webcam.
🚀 FeaturesReal-time Hand Tracking: 
Powered by Google's MediaPipe for sub-millisecond gesture detection.
Dynamic Difficulty: The game speed increases every 5 points, challenging your reflexes.
Smooth Physics: Implements "Lerp" (Linear Interpolation) for fluid bird movement that follows your finger naturally.
Visual Effects: Includes a particle explosion system upon collision, parallax scrolling clouds, and a stylized "AR" (Augmented Reality) camera overlay.
🛠️ Tech StackPython
3.xOpenCV: For camera feed handling and UI rendering.
MediaPipe: For high-fidelity hand landmark detection
.NumPy: For coordinate calculations and image processing.
📦 InstallationClone the Repository:Bashgit clone https://github.com/yourusername/gesture-flappy.git
cd gesture-flappy
Install Dependencies:Make sure you have pip installed, then run:Bashpip install opencv-python mediapipe numpy
Run the Game:Bashpython flappy_hand.py
🎮 How to Play1.
PositioningSit at a comfortable distance from your webcam.
Ensure your hand is visible within the frame.
2 ControlsMove Bird: Move your Index Finger up and down. 
The bird tracks Landmark #8 (the tip of your index finger).
Spacebar: Start the game from the menu or pause while playing.
R Key: Restart the game after a "Game Over."Q Key: Quit the game.
3. ObjectiveNavigate the bird through the gaps in the pipes. 
If you hit a pipe or the ground, the game is over!
🧠 How it WorksThe game uses the Hand Landmarks model to identify 21 3D landmarks on a hand.
Specifically, we extract the normalized $y$ coordinate of the index finger tip.
To ensure a smooth experience, the bird's position ($y_{bird}$) is updated using a smoothing factor ($\alpha$):$$y_{bird} = y_{bird} + (y_{target} - y_{bird}) \cdot \alpha$$Where $\alpha$ (SMOOTHING) is set to approximately 0.25 to filter out camera jitter.
