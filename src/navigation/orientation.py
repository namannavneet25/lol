import os
import json
import math
import numpy as np

class PhoneOrientationEstimator:
    """
    Automatic Phone-to-Vehicle Orientation & Gravity Alignment Engine (Sections 7-8 of plan.pdf).
    
    Solves unmounted/handheld phone issues:
    1. Gravity Vector Tracking: Tracks the true vertical axis by low-pass filtering 3D accelerometer data.
    2. True Vertical Yaw Rate: Projects 3D angular velocity onto the gravity unit vector:
         omega_yaw = omega · g_hat
       This isolates the true turn/yaw rotation even if the phone is tilted at arbitrary angles.
    3. Horizontal Acceleration: Removes the 1g gravity component and projects acceleration onto the horizontal plane:
         a_horiz = a - (a · g_hat) * g_hat
    4. Vehicle Forward Yaw Alignment: Estimates azimuth offset between phone heading and vehicle forward using GNSS course.
    """
    def __init__(self, alpha_gravity=0.05, gyro_bias=(0.0, 0.0, 0.0)):
        # Low-pass filter coefficient for gravity vector estimation
        self.alpha_g = alpha_gravity
        # Estimated gravity vector in phone frame (initially assuming Z up/down)
        self.g_vec = np.array([0.0, 0.0, 9.81], dtype=np.float64)
        self.is_initialized = False
        self.gyro_bias = np.array(gyro_bias, dtype=np.float64)
        
        # Pre-Drive Stand Leveling & Coordinate System Lock
        self.is_stand_calibrated = False
        self.R_level = np.eye(3, dtype=np.float64)
        self.last_tilt_rate = 0.0
        
        # Vehicle forward alignment angle offset theta: psi_vehicle = psi_phone + theta
        self.forward_yaw_offset = 0.0
        self.yaw_aligned = False
        self.consecutive_align_samples = 0

    def compute_leveling_matrix(self, g_vec):
        """Computes the 3x3 orthonormal leveling matrix R_level from stationary resting gravity."""
        g_norm = np.linalg.norm(g_vec)
        if g_norm < 1e-4:
            z_level = np.array([0.0, 0.0, 1.0])
        else:
            z_level = g_vec / g_norm
            
        y_ref = np.array([0.0, 1.0, 0.0])
        y_proj = y_ref - np.dot(y_ref, z_level) * z_level
        y_norm = np.linalg.norm(y_proj)
        if y_norm < 1e-3:
            x_ref = np.array([1.0, 0.0, 0.0])
            x_proj = x_ref - np.dot(x_ref, z_level) * z_level
            x_level = x_proj / np.linalg.norm(x_proj)
            y_level = np.cross(z_level, x_level)
            y_level = y_level / np.linalg.norm(y_level)
        else:
            y_level = y_proj / y_norm
            x_level = np.cross(y_level, z_level)
            x_level = x_level / np.linalg.norm(x_level)
            
        self.R_level = np.vstack([x_level, y_level, z_level])
        self.g_vec = z_level * 9.81
        self.is_stand_calibrated = True
        self.is_initialized = True
        return self.R_level

    def load_calibration(self, filepath_or_dict):
        """Loads calibration parameters exported by web calibrator."""
        import json
        if isinstance(filepath_or_dict, (str, bytes, os.PathLike)):
            if not os.path.exists(filepath_or_dict):
                return False
            with open(filepath_or_dict, "r") as f:
                data = json.load(f)
        else:
            data = filepath_or_dict

        if "gyro_bias_rad_s" in data:
            b = data["gyro_bias_rad_s"]
            self.gyro_bias = np.array([b.get("gx", 0.0), b.get("gy", 0.0), b.get("gz", 0.0)], dtype=np.float64)
        if "resting_gravity_vector" in data:
            g = data["resting_gravity_vector"]
            self.g_vec = np.array([g.get("x", 0.0), g.get("y", 0.0), g.get("z", 9.81)], dtype=np.float64)
            self.compute_leveling_matrix(self.g_vec)
        if "rotation_matrix" in data:
            self.R_level = np.array(data["rotation_matrix"], dtype=np.float64)
            self.is_stand_calibrated = True
        if "mounting_yaw_offset_deg" in data:
            self.forward_yaw_offset = math.radians(data["mounting_yaw_offset_deg"])
            self.yaw_aligned = True
        return True

    def update(self, ax, ay, az, gx, gy, gz):
        """
        Updates orientation estimates given raw 3D IMU sample.
        
        Returns:
          true_yaw_rate (rad/s): Rotation rate around true vertical (gravity axis).
          a_horiz_mag (m/s^2): Horizontal acceleration magnitude (gravity removed).
          pitch_deg (float): Estimated phone pitch angle.
          roll_deg (float): Estimated phone roll angle.
        """
        a = np.array([ax, ay, az], dtype=np.float64)
        omega = np.array([
            gx - self.gyro_bias[0],
            gy - self.gyro_bias[1],
            gz - self.gyro_bias[2]
        ], dtype=np.float64)
        
        # Pre-Drive Stand Calibrated Mode: Rock-solid fixed leveling transformation
        if self.is_stand_calibrated:
            a_level = self.R_level @ a
            omega_level = self.R_level @ omega
            true_yaw_rate = float(omega_level[2])
            a_horiz_mag = float(np.hypot(a_level[0], a_level[1]))
            self.last_tilt_rate = float(np.hypot(omega_level[0], omega_level[1]))
            
            # Rotate level accelerations by mounting azimuth offset theta:
            theta = self.forward_yaw_offset
            # a_level[1] is along phone top horizontal projection (forward), a_level[0] is lateral right
            a_veh_fwd = float(math.sin(theta) * a_level[0] + math.cos(theta) * a_level[1])
            a_veh_lat = float(math.cos(theta) * a_level[0] - math.sin(theta) * a_level[1])

            # Attitude angles from locked leveling matrix
            g_hat = self.R_level[2]
            pitch_deg = math.degrees(math.atan2(g_hat[0], math.hypot(g_hat[1], g_hat[2])))
            roll_deg = math.degrees(math.atan2(g_hat[1], g_hat[2]))
            return true_yaw_rate, a_veh_fwd, a_veh_lat, a_horiz_mag, pitch_deg, roll_deg

        # Uncalibrated Dynamic Preview Mode: Low-Pass Filter
        if not self.is_initialized:
            a_mag = np.linalg.norm(a)
            if 8.0 < a_mag < 11.5:
                self.g_vec = a.copy()
                self.is_initialized = True
        else:
            a_mag = np.linalg.norm(a)
            omega_mag = np.linalg.norm(omega)
            # Only adapt gravity when linear acceleration is near 1g and not spinning/wobbling rapidly (< 5.7 deg/s)
            if 8.5 < a_mag < 11.0 and omega_mag < 0.1:
                self.g_vec = (1.0 - self.alpha_g) * self.g_vec + self.alpha_g * a
                
        g_norm = np.linalg.norm(self.g_vec)
        if g_norm < 1e-4:
            g_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            g_hat = self.g_vec / g_norm
            
        # 2. Extract True Vertical Yaw Rate (Projection of angular velocity onto gravity axis)
        true_yaw_rate = float(np.dot(omega, g_hat))
        omega_sq = float(np.dot(omega, omega))
        self.last_tilt_rate = math.sqrt(max(0.0, omega_sq - true_yaw_rate**2))
        
        # 3. Horizontal Acceleration (Gravity Removal)
        a_vertical_comp = np.dot(a, g_hat) * g_hat
        a_horiz = a - a_vertical_comp
        a_horiz_mag = float(np.linalg.norm(a_horiz))

        # Project along phone top horizontal projection
        y_proj = np.array([0.0, 1.0, 0.0]) - g_hat[1] * g_hat
        y_norm = np.linalg.norm(y_proj)
        if y_norm > 1e-3:
            y_u = y_proj / y_norm
            x_u = np.cross(y_u, g_hat)
            x_u = x_u / np.linalg.norm(x_u)
            a_fwd_raw = float(np.dot(a_horiz, y_u))
            a_lat_raw = float(np.dot(a_horiz, x_u))
        else:
            a_fwd_raw = float(a_horiz[1])
            a_lat_raw = float(a_horiz[0])

        theta = self.forward_yaw_offset
        a_veh_fwd = float(math.sin(theta) * a_lat_raw + math.cos(theta) * a_fwd_raw)
        a_veh_lat = float(math.cos(theta) * a_lat_raw - math.sin(theta) * a_fwd_raw)
        
        # 4. Compute Roll and Pitch angles for diagnostics
        pitch = math.atan2(g_hat[0], math.sqrt(g_hat[1]**2 + g_hat[2]**2))
        roll = math.atan2(g_hat[1], g_hat[2])
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)
        
        return true_yaw_rate, a_veh_fwd, a_veh_lat, a_horiz_mag, pitch_deg, roll_deg

    def align_with_gnss_course(self, current_phone_yaw_rad, gnss_course_rad, vehicle_speed_mps):
        """
        Aligns phone yaw with vehicle forward direction using GNSS velocity vector (Section 8 of plan.pdf).
        Only activates when moving fast enough (> 2.0 m/s) to ensure GNSS heading validity.
        """
        if vehicle_speed_mps > 2.0:
            diff = (gnss_course_rad - current_phone_yaw_rad + math.pi) % (2 * math.pi) - math.pi
            if not self.yaw_aligned:
                self.forward_yaw_offset = diff
                self.yaw_aligned = True
            else:
                self.forward_yaw_offset = (1.0 - 0.05) * self.forward_yaw_offset + 0.05 * diff
            self.consecutive_align_samples += 1
            return True
        return False
