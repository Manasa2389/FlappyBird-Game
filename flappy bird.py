import cv2
import mediapipe as mp
import numpy as np
import random
import math
import sys

class Particle:
    """A simple particle class for explosion effects."""
    # FIX: Corrected _init_ to __init__
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 1.0  # 1.0 = 100% opacity/life
        self.decay = random.uniform(0.02, 0.05)
        self.size = random.randint(3, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        return self.life > 0

    def draw(self, img):
        if self.life > 0:
            # Fade out by reducing radius logic or color (simulated here by size)
            # Ensure color is in (B, G, R) format for OpenCV
            current_size = int(self.size * self.life)
            
            # Create a tuple for the color (B, G, R)
            bgr_color = (self.color[0], self.color[1], self.color[2])
            
            if current_size > 0:
                cv2.circle(img, (int(self.x), int(self.y)), current_size, bgr_color, -1)

class FlappyHandGame:
    # FIX: Corrected _init_ to __init__
    def __init__(self):
        # Screen settings
        self.width = 640
        self.height = 480
        
        # Game Colors (B, G, R) - Ensure colors are BGR for OpenCV
        self.COLOR_SKY = (135, 206, 235)      # Sky Blue
        self.COLOR_PIPE = (50, 205, 50)      # Lime Green
        self.COLOR_PIPE_DARK = (34, 139, 34)
        self.COLOR_PIPE_LIGHT = (144, 238, 144)
        self.COLOR_BIRD = (255, 215, 0)      # Gold (B=0, G=215, R=255 -> OpenCV BGR is 255, 215, 0)
        self.COLOR_BEAK = (255, 140, 0)      # Orange
        self.COLOR_WING = (255, 240, 200)    # Light Yellow/White
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GROUND = (222, 203, 118)  # Sandy/Ground color
        
        # Game Constants
        self.BIRD_X = 100
        self.BIRD_RADIUS = 20
        self.PIPE_WIDTH = 70
        self.PIPE_GAP = 170
        self.BASE_SPEED = 6
        self.SMOOTHING = 0.25 

        # Setup Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam. Check connectivity and permissions.")
            # We don't exit here, but the loop needs to handle the failure.
            # However, for a fully working script, we must rely on a working camera.
            # Added sys.exit() for robust failure, otherwise the traceback will still occur in run().
            self.cap.release()
            sys.exit("Webcam failed to open.")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Setup MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Animation State
        self.wing_frame = 0
        self.clouds = self._generate_clouds()
        self.particles = []
        self.flash_timer = 0
        self.ground_offset = 0
        
        # Game State
        self.reset_game()

    def _generate_clouds(self):
        """Creates random initial clouds."""
        clouds = []
        for _ in range(5):
            clouds.append({
                'x': random.randint(0, self.width),
                'y': random.randint(20, self.height // 2),
                'speed': random.uniform(1, 3),
                'size': random.randint(20, 50)
            })
        return clouds

    def reset_game(self):
        """Resets all game variables to start state."""
        self.bird_y = self.height // 2
        self.prev_bird_y = self.bird_y # For calculating tilt
        self.pipes = [] # List of [x_pos, bottom_pipe_height]
        self.score = 0
        self.game_active = False
        self.game_over = False
        self.frame_count = 0
        self.particles = []
        self.current_speed = self.BASE_SPEED

    def get_finger_position(self, img):
        """Detects hand and returns the Y position of the index finger tip."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # NOTE: MediaPipe requires the RGB image, but the draw landmarks should happen on the BGR image (img)
        # to ensure the landmarks are visible on the camera feed.
        results = self.hands.process(img_rgb)
        
        target_y = None
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Draw skeleton on screen (on the original BGR image)
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                
                # Get Index Finger Tip (Landmark 8)
                y_norm = hand_lms.landmark[8].y
                target_y = int(y_norm * self.height)
                
        return target_y

    def update_physics(self, target_y):
        """Moves the bird and handles animation physics."""
        self.prev_bird_y = self.bird_y
        
        if target_y is not None:
            # Smooth movement
            self.bird_y = int(self.bird_y + (target_y - self.bird_y) * self.SMOOTHING)
            # Clamp to screen, avoiding the ground area (last 40 pixels)
            self.bird_y = max(self.BIRD_RADIUS, min(self.height - 40 - self.BIRD_RADIUS, self.bird_y))
        
        # If no hand is detected, the bird slowly drifts downwards slightly (Optional Gravity)
        # if target_y is None and self.game_active:
        #    self.bird_y = min(self.bird_y + 3, self.height - 40 - self.BIRD_RADIUS)


    def generate_pipes(self):
        """Adds new pipes at regular intervals."""
        self.frame_count += 1
        # Difficulty scaling: Distance between pipes
        spawn_rate = 50 if self.score > 10 else 60
        
        if self.frame_count % spawn_rate == 0:
            min_height = 80
            max_height = self.height - self.PIPE_GAP - 80
            bottom_h = random.randint(min_height, max_height)
            self.pipes.append([self.width, bottom_h])

    def update_game_objects(self):
        """Moves pipes, clouds, ground, and checks score."""
        # Update Pipes
        for pipe in self.pipes:
            pipe[0] -= self.current_speed
        
        # Remove off-screen pipes
        if self.pipes and self.pipes[0][0] < -self.PIPE_WIDTH:
            self.pipes.pop(0)
            self.score += 1
            self.flash_timer = 3 # Flash screen for 3 frames
            
            # Difficulty Scaling: Speed up every 5 points
            if self.score % 5 == 0:
                self.current_speed += 1

        # Update Clouds (Parallax)
        for cloud in self.clouds:
            cloud['x'] -= cloud['speed'] * (0.5 if self.game_active else 0.2)
            if cloud['x'] < -100:
                cloud['x'] = self.width + 100
                cloud['y'] = random.randint(20, self.height // 2)

        # Update Ground
        self.ground_offset = (self.ground_offset + self.current_speed) % 40

    def check_collisions(self):
        """Checks if bird hit a pipe or ground."""
        # 1. Ground Collision
        if self.bird_y >= self.height - 40 - self.BIRD_RADIUS:
            return True

        # 2. Pipe Collision
        for pipe in self.pipes:
            pipe_x = pipe[0]
            bottom_h = pipe[1]
            top_bottom_edge = bottom_h - self.PIPE_GAP
            
            # Hitbox slightly smaller than visual bird for fairness
            hitbox_r = self.BIRD_RADIUS - 4 

            # Check X alignment
            if self.BIRD_X + hitbox_r > pipe_x and self.BIRD_X - hitbox_r < pipe_x + self.PIPE_WIDTH:
                # Check Y alignment
                if self.bird_y - hitbox_r < top_bottom_edge or self.bird_y + hitbox_r > bottom_h:
                    return True
        return False

    def explode_particles(self):
        """Spawns particles at bird location."""
        # Particles use BGR color for consistency
        bgr_color = (255, 215, 0) # Gold
        for _ in range(25):
            self.particles.append(Particle(self.BIRD_X, self.bird_y, bgr_color))

    def draw_background(self, img):
        """Draws sky, moving clouds, and scrolling ground."""
        # Sky
        img[:] = self.COLOR_SKY

        # Clouds
        for cloud in self.clouds:
            cx, cy, r = int(cloud['x']), int(cloud['y']), int(cloud['size'])
            # Draw fluffy cloud (3 overlapping circles)
            cv2.circle(img, (cx, cy), r, (255, 255, 255), -1)
            cv2.circle(img, (cx - r//2, cy + r//4), r//2 + 5, (255, 255, 255), -1)
            cv2.circle(img, (cx + r//2, cy + r//4), r//2 + 5, (255, 255, 255), -1)

        # Ground (Scrolling stripes)
        ground_y = self.height - 40
        cv2.rectangle(img, (0, ground_y), (self.width, self.height), self.COLOR_GROUND, -1)
        # Ground top border (Darker line)
        cv2.line(img, (0, ground_y), (self.width, ground_y), (45, 60, 85), 3) 
        
        # Ground stripes logic
        for i in range(-40, self.width, 40):
            x = i - int(self.ground_offset)
            pt1 = (x, ground_y)
            pt2 = (x + 20, self.height)
            # Clip drawing to avoid errors if points are way off (though cv2 handles it usually)
            cv2.line(img, pt1, pt2, (235, 215, 130), 2) # Light sand color for stripes

    def draw_pipes(self, img):
        """Draws stylized pipes."""
        for pipe in self.pipes:
            x, bottom_h = int(pipe[0]), int(pipe[1])
            top_bottom_edge = bottom_h - self.PIPE_GAP
            
            # --- Top Pipe ---
            # Main body
            cv2.rectangle(img, (x, 0), (x + self.PIPE_WIDTH, top_bottom_edge), self.COLOR_PIPE, -1)
            # Highlight (Left side light)
            cv2.rectangle(img, (x, 0), (x + 10, top_bottom_edge), self.COLOR_PIPE_LIGHT, -1)
            # Shadow (Right side dark)
            cv2.rectangle(img, (x + self.PIPE_WIDTH - 5, 0), (x + self.PIPE_WIDTH, top_bottom_edge), self.COLOR_PIPE_DARK, -1)
            # Border
            cv2.rectangle(img, (x, 0), (x + self.PIPE_WIDTH, top_bottom_edge), self.COLOR_PIPE_DARK, 2)
            # Cap
            cap_h = 25
            cv2.rectangle(img, (x - 4, top_bottom_edge - cap_h), (x + self.PIPE_WIDTH + 4, top_bottom_edge), self.COLOR_PIPE, -1)
            cv2.rectangle(img, (x - 4, top_bottom_edge - cap_h), (x + self.PIPE_WIDTH + 4, top_bottom_edge), self.COLOR_PIPE_DARK, 2)

            # --- Bottom Pipe ---
            # Main body
            cv2.rectangle(img, (x, bottom_h), (x + self.PIPE_WIDTH, self.height - 40), self.COLOR_PIPE, -1)
            # Highlight
            cv2.rectangle(img, (x, bottom_h), (x + 10, self.height - 40), self.COLOR_PIPE_LIGHT, -1)
            # Shadow
            cv2.rectangle(img, (x + self.PIPE_WIDTH - 5, bottom_h), (x + self.PIPE_WIDTH, self.height - 40), self.COLOR_PIPE_DARK, -1)
            # Border
            cv2.rectangle(img, (x, bottom_h), (x + self.PIPE_WIDTH, self.height - 40), self.COLOR_PIPE_DARK, 2)
            # Cap
            cv2.rectangle(img, (x - 4, bottom_h), (x + self.PIPE_WIDTH + 4, bottom_h + cap_h), self.COLOR_PIPE, -1)
            cv2.rectangle(img, (x - 4, bottom_h), (x + self.PIPE_WIDTH + 4, bottom_h + cap_h), self.COLOR_PIPE_DARK, 2)

    def draw_bird(self, img):
        """Draws animated bird with tilt and flapping wings."""
        # Calculate Tilt based on vertical movement
        dy = self.bird_y - self.prev_bird_y
        tilt_offset = int(np.clip(dy * 2, -15, 15)) # Pixel offset for "looking up/down"

        # 1. Body
        cv2.circle(img, (self.BIRD_X, self.bird_y), self.BIRD_RADIUS, self.COLOR_BIRD, -1)
        cv2.circle(img, (self.BIRD_X, self.bird_y), self.BIRD_RADIUS, (0,0,0), 2)

        # 2. Eye (Moves with tilt)
        eye_y = self.bird_y - 6 + (tilt_offset // 2)
        cv2.circle(img, (self.BIRD_X + 8, eye_y), 8, (255,255,255), -1)
        cv2.circle(img, (self.BIRD_X + 10, eye_y), 3, (0,0,0), -1)

        # 3. Beak (Moves with tilt)
        beak_y = self.bird_y + 5 + (tilt_offset // 2)
        beak_pts = np.array([
            [self.BIRD_X + 15, beak_y],
            [self.BIRD_X + 35, beak_y + 5],
            [self.BIRD_X + 15, beak_y + 10]
        ], np.int32)
        cv2.fillPoly(img, [beak_pts], self.COLOR_BEAK)
        cv2.polylines(img, [beak_pts], True, (0,0,0), 1)

        # 4. Wing (Flapping Animation)
        # Sine wave for wing flapping frequency
        self.wing_frame += 0.4
        flap_y = int(6 * math.sin(self.wing_frame))
        
        wing_y_pos = self.bird_y + 5 + flap_y
        # Draw a small ellipse for the wing
        cv2.ellipse(img, (self.BIRD_X - 6, wing_y_pos), (10, 7), 0, 0, 360, self.COLOR_WING, -1)
        cv2.ellipse(img, (self.BIRD_X - 6, wing_y_pos), (10, 7), 0, 0, 360, (0,0,0), 1)

    def run(self):
        print("Game Running... Use your index finger to control the bird's vertical position.")
        print("Press 'Space' to Start/Stop, 'R' to Restart, 'Q' to Quit")
        
        while True:
            success, img = self.cap.read()
            if not success or img is None:
                print("Warning: Failed to read frame from camera.")
                # Allow user to quit if camera fails completely
                if cv2.waitKey(1) == ord('q'): break
                continue
            
            # Prepare Frame
            img = cv2.flip(img, 1) # Flip horizontally for selfie view
            
            # Get Finger Position & Draw Landmarks (on BGR image 'img')
            target_y = self.get_finger_position(img)
            
            # Store a copy of the camera feed with landmarks for blending
            camera_feed_with_landmarks = img.copy()
            
            # Draw base visuals (Game environment)
            self.draw_background(img)
            
            # Blend camera feed lightly so you can see your hand but focus on game
            # This makes the game "AR" by showing the user their tracked hand within the game scene.
            cv2.addWeighted(camera_feed_with_landmarks, 0.25, img, 0.75, 0, img)

            # --- Game States ---
            if not self.game_active and not self.game_over:
                # MENU
                self.update_physics(target_y)
                self.draw_bird(img)
                
                # Update background elements even in menu
                for cloud in self.clouds: cloud['x'] -= 0.5
                
                # Menu Text
                cv2.putText(img, "FLAPPY GESTURE", (120, 180), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 4)
                cv2.putText(img, "FLAPPY GESTURE", (120, 180), cv2.FONT_HERSHEY_DUPLEX, 1.5, self.COLOR_PIPE_LIGHT, 2)
                
                cv2.putText(img, "Show Index Finger to Control Bird", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(img, "Press SPACE to Start", (170, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            elif self.game_active:
                # PLAY
                self.update_physics(target_y)
                self.generate_pipes()
                self.update_game_objects()
                
                self.draw_pipes(img)
                self.draw_bird(img)
                
                # Check Death
                if self.check_collisions():
                    self.game_active = False
                    self.game_over = True
                    self.explode_particles()

                # Score Flash
                if self.flash_timer > 0:
                    overlay = img.copy()
                    overlay[:] = (255, 255, 255)
                    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                    self.flash_timer -= 1
                
                # Draw Score
                cv2.putText(img, str(self.score), (self.width//2 - 15, 80), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,0,0), 5)
                cv2.putText(img, str(self.score), (self.width//2 - 15, 80), cv2.FONT_HERSHEY_DUPLEX, 2.5, self.COLOR_BEAK, 2)

            elif self.game_over:
                # GAME OVER
                self.draw_pipes(img) # Draw frozen pipes
                
                # Update and Draw Particles
                self.particles = [p for p in self.particles if p.update()]
                for p in self.particles:
                    p.draw(img)
                
                # Draw Dead Bird (Fall to ground)
                if self.bird_y < self.height - 40 - self.BIRD_RADIUS:
                    self.bird_y += 10 # Fall to ground
                self.prev_bird_y = self.bird_y - 10 # Force "down" look
                self.draw_bird(img)

                # Text
                cv2.putText(img, "GAME OVER", (180, 200), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 150), 3)
                cv2.putText(img, f"Score: {self.score}", (240, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, "Press R to Restart", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # --- Input Handling ---
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 32: # Space
                if not self.game_active and not self.game_over:
                    self.game_active = True
                elif self.game_active:
                    # Optional: Pause game on space
                    self.game_active = False
            if key == ord('r'):
                if self.game_over:
                    self.reset_game()

            cv2.imshow("Gesture Flappy Bird", img)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = FlappyHandGame()
    game.run()