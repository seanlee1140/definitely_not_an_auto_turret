"""
Ball Trajectory Tracker
========================

Kalman filter based 3D ball tracker and trajectory predictor.
Takes noisy 3D position measurements from stereo triangulation 
and estimates position, velocity, and acceleration. Predicts 
future ball positions for intercept calculation.

State vector (9 elements):
    [x, y, z, vx, vy, vz, ax, ay, az]

Measurements (3 elements):
    [x, y, z] from stereo triangulation

Coordinate system (relative to left camera):
    X = right, Y = down, Z = forward

Dependencies:
    - numpy
    - opencv (cv2)
"""

import numpy as np
import cv2


class BallTracker:
    """
    Kalman filter for 3D ball tracking with trajectory prediction.

    Args:
        dt: Time step between frames in seconds (0.033 ≈ 30fps).
        process_noise: How much we expect the model to deviate.
            Higher = more responsive but noisier.
        measurement_noise: How noisy our triangulation measurements are.
            Higher = smoother but slower to react.
        gravity_mms2: Gravity in mm/s². Default 9810 (9.81 m/s²).
            Set negative or positive depending on your Y-axis direction.
            Default assumes Y-down (positive Y = downward).
    """

    def __init__(self, dt=0.033, process_noise=1e-2, measurement_noise=5.0,
                 gravity_mms2=9810.0):
        self.dt = dt
        self.gravity = np.array([0, gravity_mms2, 0], dtype=np.float32)

        self.kf = cv2.KalmanFilter(9, 3)

        dt2 = 0.5 * dt * dt
        self.kf.transitionMatrix = np.eye(9, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 3] = dt
            self.kf.transitionMatrix[i, i + 6] = dt2
            self.kf.transitionMatrix[i + 3, i + 6] = dt

        self.kf.measurementMatrix = np.zeros((3, 9), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 2] = 1

        self.kf.processNoiseCov = np.eye(9, dtype=np.float32) * process_noise
        self.kf.processNoiseCov[6, 6] = 1.0
        self.kf.processNoiseCov[7, 7] = 1.0
        self.kf.processNoiseCov[8, 8] = 1.0

        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 100.0

        self.initialized = False
        self.last_update_time = None
        self.measurements = []

    def update(self, position_3d, timestamp=None):
        """
        Feed a new 3D measurement into the filter.

        Args:
            position_3d: numpy array [x, y, z] in mm from triangulation.
            timestamp: Optional time in seconds. If provided, dt is 
                computed from the previous update for better accuracy.

        Returns:
            Filtered [x, y, z] position estimate in mm.
        """
        if timestamp is not None and self.last_update_time is not None:
            actual_dt = timestamp - self.last_update_time
            if actual_dt > 0:
                self._update_dt(actual_dt)
        self.last_update_time = timestamp

        measurement = np.array(position_3d, dtype=np.float32).reshape(3, 1)

        if not self.initialized:
            self.kf.statePost = np.zeros((9, 1), dtype=np.float32)
            self.kf.statePost[0] = measurement[0]
            self.kf.statePost[1] = measurement[1]
            self.kf.statePost[2] = measurement[2]
            self.initialized = True
            self.measurements.append(position_3d.copy())
            return position_3d

        self.kf.predict()
        corrected = self.kf.correct(measurement)
        self.measurements.append(position_3d.copy())

        return corrected[:3].flatten()

    def predict_future_position(self, t_seconds):
        """
        Predict where the ball will be t_seconds from now.

        Uses current estimated velocity and acceleration, plus gravity,
        with kinematic equation: p = p0 + v*t + 0.5*a*t²

        Args:
            t_seconds: How far into the future to predict (seconds).

        Returns:
            numpy array [x, y, z] predicted position in mm, or None 
            if the filter hasn't been initialized.
        """
        if not self.initialized:
            return None

        state = self.kf.statePost.flatten()
        pos = state[0:3]
        vel = state[3:6]
        acc = state[6:9]

        total_acc = acc + self.gravity
        future_pos = pos + vel * t_seconds + 0.5 * total_acc * t_seconds**2

        return future_pos

    def get_velocity(self):
        """Return current estimated velocity [vx, vy, vz] in mm/s."""
        if not self.initialized:
            return None
        return self.kf.statePost[3:6].flatten()

    def get_speed(self):
        """Return current estimated speed (scalar) in mm/s."""
        vel = self.get_velocity()
        if vel is None:
            return 0.0
        return float(np.linalg.norm(vel))

    def reset(self):
        """Reset the tracker for a new ball / new rally."""
        self.initialized = False
        self.last_update_time = None
        self.measurements = []
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 100.0

    def _update_dt(self, new_dt):
        """Update transition matrix when dt changes between frames."""
        dt = new_dt
        dt2 = 0.5 * dt * dt
        self.kf.transitionMatrix = np.eye(9, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 3] = dt
            self.kf.transitionMatrix[i, i + 6] = dt2
            self.kf.transitionMatrix[i + 3, i + 6] = dt