/**
 * OrientationCalibrator
 * Client-side JavaScript implementation of the Pre-Drive Stand & Mount Calibration Engine
 * Matches Python implementation in src/navigation/orientation.py
 */

class OrientationCalibrator {
    constructor(alphaGravity = 0.05) {
        this.alphaG = alphaGravity;
        this.gVec = [0.0, 0.0, 9.81];
        this.isInitialized = false;

        // Pre-Drive Stand Calibration Lock
        this.isStandLocked = false;
        this.R_level = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];

        // Calibrated zero-rate gyro biases (rad/s)
        this.gyroBias = [0.0, 0.0, 0.0];
        this.isCalibrated = false;

        // Car forward mounting azimuth offset (degrees)
        this.mountingYawOffsetDeg = 0.0;
        this.isMountingAligned = false;

        // Calibration session state
        this.isCalibrating = false;
        this.calibrationSamples = [];
        this.calibrationTargetCount = 180; // 3.0 seconds at ~60Hz
        this.onCalibrationProgress = null;
        this.onCalibrationComplete = null;
    }

    /**
     * Computes orthonormal 3x3 leveling matrix from stationary gravity vector
     */
    computeLevelingMatrix(gx, gy, gz) {
        const gNorm = Math.hypot(gx, gy, gz);
        const zL = gNorm > 1e-4 ? [gx / gNorm, gy / gNorm, gz / gNorm] : [0.0, 0.0, 1.0];

        // Reference vector: portrait top y = [0, 1, 0]
        const yDotZ = zL[1];
        let yProj = [0.0 - yDotZ * zL[0], 1.0 - yDotZ * zL[1], 0.0 - yDotZ * zL[2]];
        let yNorm = Math.hypot(...yProj);

        let xL, yL;
        if (yNorm < 1e-3) {
            // Lying flat, reference x = [1, 0, 0]
            const xDotZ = zL[0];
            let xProj = [1.0 - xDotZ * zL[0], 0.0 - xDotZ * zL[1], 0.0 - xDotZ * zL[2]];
            let xNorm = Math.hypot(...xProj);
            xL = [xProj[0] / xNorm, xProj[1] / xNorm, xProj[2] / xNorm];
            yL = [
                zL[1] * xL[2] - zL[2] * xL[1],
                zL[2] * xL[0] - zL[0] * xL[2],
                zL[0] * xL[1] - zL[1] * xL[0]
            ];
        } else {
            yL = [yProj[0] / yNorm, yProj[1] / yNorm, yProj[2] / yNorm];
            xL = [
                yL[1] * zL[2] - yL[2] * zL[1],
                yL[2] * zL[0] - yL[0] * zL[2],
                yL[0] * zL[1] - yL[1] * zL[0]
            ];
        }

        this.R_level = [xL, yL, zL];
        this.isStandLocked = true;
        this.gVec = [zL[0] * 9.81, zL[1] * 9.81, zL[2] * 9.81];
        return this.R_level;
    }

    /**
     * Processes a single raw IMU frame from DeviceMotionEvent
     */
    update(ax, ay, az, gx, gy, gz) {
        // Subtract calibrated zero-rate bias
        const gxCorr = gx - this.gyroBias[0];
        const gyCorr = gy - this.gyroBias[1];
        const gzCorr = gz - this.gyroBias[2];

        const aMag = Math.hypot(ax, ay, az);

        // Accumulate samples during stationary calibration session
        if (this.isCalibrating) {
            this.calibrationSamples.push({
                gx, gy, gz,
                ax, ay, az,
                aMag
            });

            const progress = this.calibrationSamples.length / this.calibrationTargetCount;
            if (this.onCalibrationProgress) {
                this.onCalibrationProgress(Math.min(1.0, progress));
            }

            if (this.calibrationSamples.length >= this.calibrationTargetCount) {
                this._finishCalibration();
            }
        }

        // --- MODE 1: Stand Leveling Locked (Post-Calibration) ---
        if (this.isStandLocked) {
            const R = this.R_level;
            // Rotate accel into level horizontal frame
            const axL = R[0][0] * ax + R[0][1] * ay + R[0][2] * az;
            const ayL = R[1][0] * ax + R[1][1] * ay + R[1][2] * az;
            const azL = R[2][0] * ax + R[2][1] * ay + R[2][2] * az;

            // Rotate gyro into level horizontal frame
            const gxL = R[0][0] * gxCorr + R[0][1] * gyCorr + R[0][2] * gzCorr;
            const gyL = R[1][0] * gxCorr + R[1][1] * gyCorr + R[1][2] * gzCorr;
            const gzL = R[2][0] * gxCorr + R[2][1] * gyCorr + R[2][2] * gzCorr;

            // Pure vehicle yaw is strictly rotation around vertical axis Z
            const trueYawRateRad = gzL;
            const trueYawRateDeg = gzL * (180.0 / Math.PI);

            // Horizontal linear acceleration (gravity completely isolated onto Z)
            const aHorizMag = Math.hypot(axL, ayL);

            // Mount vibrations and bumps (orthogonal to gravity in horizontal plane)
            const tiltRateRad = Math.hypot(gxL, gyL);
            const tiltRateDeg = tiltRateRad * (180.0 / Math.PI);

            const zL = R[2];
            const pitchRad = Math.atan2(zL[0], Math.hypot(zL[1], zL[2]));
            const rollRad = Math.atan2(zL[1], zL[2]);
            const pitchDeg = pitchRad * (180.0 / Math.PI);
            const rollDeg = rollRad * (180.0 / Math.PI);

            // Calculate vehicle forward and lateral accelerations with mounting azimuth offset
            const thetaRad = this.mountingYawOffsetDeg * (Math.PI / 180.0);
            const aFwd = Math.sin(thetaRad) * axL + Math.cos(thetaRad) * ayL;
            const aLat = Math.cos(thetaRad) * axL - Math.sin(thetaRad) * ayL;

            return {
                trueYawRateRad,
                trueYawRateDeg,
                tiltRateRad,
                tiltRateDeg,
                rawGxDeg: gxCorr * (180.0 / Math.PI),
                rawGyDeg: gyCorr * (180.0 / Math.PI),
                rawGzDeg: gzCorr * (180.0 / Math.PI),
                aHorizMag,
                aFwd,
                aLat,
                aRawMag: aMag,
                gMagnitude: 9.81,
                gVector: [zL[0], zL[1], zL[2]],
                pitchDeg,
                rollDeg,
                isCalibrated: this.isCalibrated,
                isStandLocked: this.isStandLocked,
                mountingYawOffsetDeg: this.mountingYawOffsetDeg
            };
        }

        // --- MODE 2: Dynamic Preview (Before Calibration) ---
        const gyroMag = Math.hypot(gxCorr, gyCorr, gzCorr);
        if (!this.isInitialized) {
            if (aMag > 7.5 && aMag < 12.0) {
                this.gVec = [ax, ay, az];
                this.isInitialized = true;
            }
        } else {
            if (aMag > 8.5 && aMag < 11.2 && gyroMag < 0.1) {
                this.gVec[0] = (1.0 - this.alphaG) * this.gVec[0] + this.alphaG * ax;
                this.gVec[1] = (1.0 - this.alphaG) * this.gVec[1] + this.alphaG * ay;
                this.gVec[2] = (1.0 - this.alphaG) * this.gVec[2] + this.alphaG * az;
            }
        }

        const gNorm = Math.hypot(...this.gVec);
        const gHat = gNorm > 1e-4 ? [this.gVec[0] / gNorm, this.gVec[1] / gNorm, this.gVec[2] / gNorm] : [0, 0, 1];

        const trueYawRateRad = gxCorr * gHat[0] + gyCorr * gHat[1] + gzCorr * gHat[2];
        const trueYawRateDeg = trueYawRateRad * (180.0 / Math.PI);

        const omegaSq = gxCorr * gxCorr + gyCorr * gyCorr + gzCorr * gzCorr;
        const tiltSq = Math.max(0.0, omegaSq - trueYawRateRad * trueYawRateRad);
        const tiltRateDeg = Math.sqrt(tiltSq) * (180.0 / Math.PI);

        const aDotG = ax * gHat[0] + ay * gHat[1] + az * gHat[2];
        const aHx = ax - aDotG * gHat[0];
        const aHy = ay - aDotG * gHat[1];
        const aHz = az - aDotG * gHat[2];
        const aHorizMag = Math.hypot(aHx, aHy, aHz);

        // Project along phone top horizontal projection
        const yDotG = gHat[1];
        const yProj = [-yDotG * gHat[0], 1.0 - yDotG * gHat[1], -yDotG * gHat[2]];
        const yNorm = Math.hypot(...yProj);
        let aFwdRaw, aLatRaw;
        if (yNorm > 1e-3) {
            const yU = [yProj[0] / yNorm, yProj[1] / yNorm, yProj[2] / yNorm];
            const xU = [
                yU[1] * gHat[2] - yU[2] * gHat[1],
                yU[2] * gHat[0] - yU[0] * gHat[2],
                yU[0] * gHat[1] - yU[1] * gHat[0]
            ];
            aFwdRaw = aHx * yU[0] + aHy * yU[1] + aHz * yU[2];
            aLatRaw = aHx * xU[0] + aHy * xU[1] + aHz * xU[2];
        } else {
            aFwdRaw = ay;
            aLatRaw = ax;
        }

        const thetaRad = this.mountingYawOffsetDeg * (Math.PI / 180.0);
        const aFwd = Math.sin(thetaRad) * aLatRaw + Math.cos(thetaRad) * aFwdRaw;
        const aLat = Math.cos(thetaRad) * aLatRaw - Math.sin(thetaRad) * aFwdRaw;

        const pitchDeg = Math.atan2(gHat[0], Math.hypot(gHat[1], gHat[2])) * (180.0 / Math.PI);
        const rollDeg = Math.atan2(gHat[1], gHat[2]) * (180.0 / Math.PI);

        return {
            trueYawRateRad,
            trueYawRateDeg,
            tiltRateRad: Math.sqrt(tiltSq),
            tiltRateDeg,
            rawGxDeg: gxCorr * (180.0 / Math.PI),
            rawGyDeg: gyCorr * (180.0 / Math.PI),
            rawGzDeg: gzCorr * (180.0 / Math.PI),
            aHorizMag,
            aFwd,
            aLat,
            aRawMag: aMag,
            gMagnitude: gNorm,
            gVector: [gHat[0], gHat[1], gHat[2]],
            pitchDeg,
            rollDeg,
            isCalibrated: this.isCalibrated,
            isStandLocked: this.isStandLocked,
            mountingYawOffsetDeg: this.mountingYawOffsetDeg
        };
    }

    /**
     * Initiates a 3-second stationary stand calibration
     */
    startStationaryCalibration(durationSec = 3.0, sampleRateHz = 60) {
        this.isCalibrating = true;
        this.calibrationSamples = [];
        this.calibrationTargetCount = Math.round(durationSec * sampleRateHz);
    }

    _finishCalibration() {
        this.isCalibrating = false;
        const N = this.calibrationSamples.length;
        if (N === 0) return;

        let sumGx = 0, sumGy = 0, sumGz = 0;
        let sumAx = 0, sumAy = 0, sumAz = 0;

        for (const s of this.calibrationSamples) {
            sumGx += s.gx;
            sumGy += s.gy;
            sumGz += s.gz;
            sumAx += s.ax;
            sumAy += s.ay;
            sumAz += s.az;
        }

        this.gyroBias = [sumGx / N, sumGy / N, sumGz / N];
        this.gVec = [sumAx / N, sumAy / N, sumAz / N];
        this.computeLevelingMatrix(this.gVec[0], this.gVec[1], this.gVec[2]);
        this.isCalibrated = true;
        this.isStandLocked = true;

        // Compute noise standard deviation
        let varGz = 0;
        for (const s of this.calibrationSamples) {
            varGz += (s.gz - this.gyroBias[2]) ** 2;
        }
        const stdGz = Math.sqrt(varGz / N);

        const result = {
            gyroBias: this.gyroBias,
            gyroBiasDeg: this.gyroBias.map(b => b * (180.0 / Math.PI)),
            gravityResting: this.gVec,
            gravityRestingMag: Math.hypot(...this.gVec),
            gyroNoiseStdDeg: stdGz * (180.0 / Math.PI),
            rotationMatrix: this.R_level,
            sampleCount: N
        };

        if (this.onCalibrationComplete) {
            this.onCalibrationComplete(result);
        }
    }

    alignMountingAzimuth(phoneHeadingDeg, vehicleCourseDeg) {
        let diff = (vehicleCourseDeg - phoneHeadingDeg + 180) % 360 - 180;
        this.mountingYawOffsetDeg = diff;
        this.isMountingAligned = true;
        return this.mountingYawOffsetDeg;
    }

    exportCalibrationJSON() {
        return {
            timestamp: new Date().toISOString(),
            is_stand_calibrated: this.isStandLocked,
            rotation_matrix: this.R_level,
            gyro_bias_rad_s: {
                gx: this.gyroBias[0],
                gy: this.gyroBias[1],
                gz: this.gyroBias[2]
            },
            resting_gravity_vector: {
                x: this.gVec[0],
                y: this.gVec[1],
                z: this.gVec[2],
                magnitude: Math.hypot(...this.gVec)
            },
            mounting_yaw_offset_deg: this.mountingYawOffsetDeg,
            is_calibrated: this.isCalibrated,
            is_mounting_aligned: this.isMountingAligned
        };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = OrientationCalibrator;
}
