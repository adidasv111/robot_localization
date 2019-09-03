/*
 * Copyright (c) 2014, 2015, 2016, Charles River Analytics, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "robot_localization/ekf.h"
#include "robot_localization/filter_common.h"

#include <XmlRpcException.h>

#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>
#include <ros/ros.h>

namespace RobotLocalization
{
  Ekf::Ekf(std::vector<double>) :
    FilterBase()  // Must initialize filter base!
  {
    controlVelocity_.resize(6);
    controlVelocity_.setZero();
    controlDelta_ = 0;
    latestControlTime_ = ros::Time::now().toSec();
    displayCounter_ = 0;
  }

  Ekf::~Ekf()
  {
  }

  void Ekf::correct(const Measurement &measurement)
  {
    FB_DEBUG("---------------------- Ekf::correct ----------------------\n" <<
             "State is:\n" << state_ << "\n"
             "Topic is:\n" << measurement.topicName_ << "\n"
             "Measurement is:\n" << measurement.measurement_ << "\n"
             "Measurement topic name is:\n" << measurement.topicName_ << "\n\n"
             "Measurement covariance is:\n" << measurement.covariance_ << "\n");

    // We don't want to update everything, so we need to build matrices that only update
    // the measured parts of our state vector. Throughout prediction and correction, we
    // attempt to maximize efficiency in Eigen.

    // First, determine how many state vector values we're updating
    std::vector<size_t> updateIndices;
    for (size_t i = 0; i < measurement.updateVector_.size(); ++i)
    {
      if (measurement.updateVector_[i])
      {
        // Handle nan and inf values in measurements
        if (std::isnan(measurement.measurement_(i)))
        {
          FB_DEBUG("Value at index " << i << " was nan. Excluding from update.\n");
        }
        else if (std::isinf(measurement.measurement_(i)))
        {
          FB_DEBUG("Value at index " << i << " was inf. Excluding from update.\n");
        } else if (i==StateMemberRr || i==StateMemberRl || i==StateMemberD)
        {
          FB_DEBUG("Value at index " << i << " was not intended to update. Excluding from update.\n");
        }
        else
        {
          updateIndices.push_back(i);
        }
      }
    }

    FB_DEBUG("Update indices are:\n" << updateIndices << "\n");

    size_t updateSize = updateIndices.size();

    // Now set up the relevant matrices
    Eigen::VectorXd stateSubset(updateSize);                              // x (in most literature)
    Eigen::VectorXd measurementSubset(updateSize);                        // z
    Eigen::MatrixXd measurementCovarianceSubset(updateSize, updateSize);  // R
    Eigen::MatrixXd stateToMeasurementSubset(updateSize, state_.rows());  // H
    Eigen::MatrixXd kalmanGainSubset(state_.rows(), updateSize);          // K
    Eigen::VectorXd innovationSubset(updateSize);                         // z - Hx

    stateSubset.setZero();
    measurementSubset.setZero();
    measurementCovarianceSubset.setZero();
    stateToMeasurementSubset.setZero();
    kalmanGainSubset.setZero();
    innovationSubset.setZero();

    // Now build the sub-matrices from the full-sized matrices (any part which isn't update corresponds to 0 in the matrices)
    for (size_t i = 0; i < updateSize; ++i)
    {
      measurementSubset(i) = measurement.measurement_(updateIndices[i]);
      stateSubset(i) = state_(updateIndices[i]);

      for (size_t j = 0; j < updateSize; ++j)
      {
        measurementCovarianceSubset(i, j) = measurement.covariance_(updateIndices[i], updateIndices[j]);
      }

      // Handle negative (read: bad) covariances in the measurement. Rather
      // than exclude the measurement or make up a covariance, just take
      // the absolute value.
      if (measurementCovarianceSubset(i, i) < 0)
      {
        FB_DEBUG("WARNING: Negative covariance for index " << i <<
                 " of measurement (value is" << measurementCovarianceSubset(i, i) <<
                 "). Using absolute value...\n");

        measurementCovarianceSubset(i, i) = ::fabs(measurementCovarianceSubset(i, i));
      }

      // If the measurement variance for a given variable is very
      // near 0 (as in e-50 or so) and the variance for that
      // variable in the covariance matrix is also near zero, then
      // the Kalman gain computation will blow up. Really, no
      // measurement can be completely without error, so add a small
      // amount in that case.
      if (measurementCovarianceSubset(i, i) < 1e-9)
      {
        FB_DEBUG("WARNING: measurement had very small error covariance for index " << updateIndices[i] <<
                 ". Adding some noise to maintain filter stability.\n");

        measurementCovarianceSubset(i, i) = 1e-9;
      }
    }

    // The state-to-measurement function, h, will now be a measurement_size x full_state_size
    // matrix, with ones in the (i, i) locations of the values to be updated
    // H is a matrix of 0s and 1s, since all the measurements sources measure the state (or parts of it) directly.
    // We don't accept raw measurement (i.e. IMU data), only measurements of the state
    for (size_t i = 0; i < updateSize; ++i)
    {
      stateToMeasurementSubset(i, updateIndices[i]) = 1;
    }
    // stateToMeasurementSubset(StateMemberRr, StateMemberRr) = 10;
    // stateToMeasurementSubset(StateMemberRl, StateMemberRl) = 10;
    // stateToMeasurementSubset(StateMemberD, StateMemberD) = 10;

    // FB_DEBUG("Current state subset is:\n" << stateSubset <<
    //          "\nMeasurement subset is:\n" << measurementSubset <<
    //          "\nMeasurement covariance subset is:\n" << measurementCovarianceSubset <<


    // (1) Compute the Kalman gain: K = (PH') / (HPH' + R)
    Eigen::MatrixXd pht = estimateErrorCovariance_ * stateToMeasurementSubset.transpose();
    Eigen::MatrixXd hphrInv  = (stateToMeasurementSubset * pht + measurementCovarianceSubset).inverse();
    kalmanGainSubset.noalias() = pht * hphrInv;

    // z - Hx
    innovationSubset = (measurementSubset - stateSubset);

    // Wrap angles in the innovation
    for (size_t i = 0; i < updateSize; ++i)
    {
      if (updateIndices[i] == StateMemberRoll  ||
          updateIndices[i] == StateMemberPitch ||
          updateIndices[i] == StateMemberYaw)
      {
        while (innovationSubset(i) < -PI)
        {
          innovationSubset(i) += TAU;
        }

        while (innovationSubset(i) > PI)
        {
          innovationSubset(i) -= TAU;
        }
      }
    }

    // (2) Check Mahalanobis distance between mapped measurement and state. true if measurement is inside threshold
    if (checkMahalanobisThreshold(innovationSubset, hphrInv, measurement.mahalanobisThresh_))
    {
      // (3) Apply the gain to the difference between the state and measurement: x = x + K(z - Hx)
      state_.noalias() += kalmanGainSubset * innovationSubset;

      // (4) Update the estimate error covariance using the Joseph form: P = (I - KH)P(I - KH)' + KRK'
      Eigen::MatrixXd gainResidual = identity_;
      gainResidual.noalias() -= kalmanGainSubset * stateToMeasurementSubset;
      estimateErrorCovariance_ = gainResidual * estimateErrorCovariance_ * gainResidual.transpose();
      estimateErrorCovariance_.noalias() += kalmanGainSubset *
                                            measurementCovarianceSubset *
                                            kalmanGainSubset.transpose();
      // Handle wrapping of angles
      wrapStateAngles();

      if (measurement.topicName_ != "odom0_twist")
      {
        std::cout << "topic name is: " << measurement.topicName_ << std::endl;
        ROS_INFO_STREAM("\nState-to-measurement subset is:\n" << stateToMeasurementSubset << "\n");
        std::cout << "update index: " << std::endl;
        for (size_t i = 0; i < updateSize; ++i)
        {
          std::cout << updateIndices[i] << " ";
        }
        std::cout << std::endl;

        ROS_INFO_STREAM("---------------------- Ekf::correct ----------------------1\n" <<
          //  "delta is " << delta << "\n" <<
            "state is " << state_ <<
            "Kalman gain subset is:\n" << kalmanGainSubset * innovationSubset << "\n");
      }

      FB_DEBUG("Kalman gain subset is:\n" << kalmanGainSubset <<
               "\nInnovation is:\n" << innovationSubset <<
               "\nCorrected full state is:\n" << state_ <<
               "\nCorrected full estimate error covariance is:\n" << estimateErrorCovariance_ <<
               "\n\n---------------------- /Ekf::correct ----------------------\n");
    }
  }

  void Ekf::predict(const double referenceTime, const double delta)
  {
    // ROS_INFO_STREAM("---------------------- Ekf::predict ----------------------1\n" <<
    //         //  "delta is " << delta << "\n" <<
    //          "state is " << state_ << "\n");
    double vl = state_(StateMemberVx) - 0.5*BaseLength*state_(StateMemberVyaw);
    double vr = state_(StateMemberVx) + 0.5*BaseLength*state_(StateMemberVyaw);
    double wl = vl/BaseRadius;
    double wr = vr/BaseRadius;

    double vx = (wr*state_(StateMemberRr) + wl*state_(StateMemberRl))/2.0;
    double vyaw = (wr*state_(StateMemberRr) - wl*state_(StateMemberRl))/state_(StateMemberD);

    double delta_yaw = vyaw*delta;
    // update state base on odometry kinematic
    state_(StateMemberX) += vx * cos(state_(StateMemberYaw)+delta_yaw/2.0) * delta;
    state_(StateMemberY) += vx * sin(state_(StateMemberYaw)+delta_yaw/2.0) * delta;
    state_(StateMemberYaw) += delta_yaw;
    // double delta_yaw = state_(StateMemberVyaw)*delta;
    // // update state base on odometry kinematic
    // state_(StateMemberX) += state_(StateMemberVx)* cos(state_(StateMemberVyaw)+delta_yaw/2.0) * delta;
    // state_(StateMemberY) += state_(StateMemberVx) * sin(state_(StateMemberVyaw)+delta_yaw/2.0) * delta;
    // state_(StateMemberYaw) += delta_yaw;

    // ROS_INFO_STREAM("---------------------- Ekf::predict ----------------------2\n" <<
    //         //  "delta is " << delta << "\n" <<
    //          "state is " << state_ << "\n");

    // We'll need these trig calculations a lot.


    double sy = ::sin(state_(StateMemberYaw));
    double cy = ::cos(state_(StateMemberYaw));

    // Much of the transfer function Jacobian is identical to the transfer function
    transferFunctionJacobian_ = transferFunction_;
    transferFunctionJacobian_(StateMemberX, StateMemberYaw) = -0.5*delta*(vl+vr)*sy;
    transferFunctionJacobian_(StateMemberX, StateMemberRl) = 0.5*delta*wl*cy;
    transferFunctionJacobian_(StateMemberX, StateMemberRr) = 0.5*delta*wr*cy;
    transferFunctionJacobian_(StateMemberY, StateMemberYaw) = 0.5*delta*(vl+vr)*cy;
    transferFunctionJacobian_(StateMemberY, StateMemberRl) = 0.5*delta*wl*sy;
    transferFunctionJacobian_(StateMemberY, StateMemberRr) = 0.5*delta*wl*sy;
    transferFunctionJacobian_(StateMemberYaw, StateMemberRl) =  -delta*wl/state_(StateMemberD);
    transferFunctionJacobian_(StateMemberYaw, StateMemberRr) =  delta*wr/state_(StateMemberD);
    transferFunctionJacobian_(StateMemberYaw, StateMemberD) = delta*((vl-vr)/(state_(StateMemberD)*state_(StateMemberD)));

    transferFunctionJacobian_(StateMemberX, StateMemberVx) = cy*delta;
    transferFunctionJacobian_(StateMemberX, StateMemberVyaw) = -sy*delta*delta/2.0;
    transferFunctionJacobian_(StateMemberY, StateMemberVx) = sy*delta;
    transferFunctionJacobian_(StateMemberY, StateMemberVyaw) = cy*delta*delta/2.0;

    FB_DEBUG("Transfer function is:\n" << transferFunction_ <<
             "\nTransfer function Jacobian is:\n" << transferFunctionJacobian_ <<
             "\nProcess noise covariance is:\n" << processNoiseCovariance_ <<
             "\nCurrent state is:\n" << state_ << "\n");

    Eigen::MatrixXd *processNoiseCovariance = &processNoiseCovariance_;

    if (useDynamicProcessNoiseCovariance_)
    {
      computeDynamicProcessNoiseCovariance(state_, delta);
      processNoiseCovariance = &dynamicProcessNoiseCovariance_;
    }

    // Handle wrapping
    wrapStateAngles();

    FB_DEBUG("Predicted state is:\n" << state_ <<
             "\nCurrent estimate error covariance is:\n" <<  estimateErrorCovariance_ << "\n");

    // (3) Project the error forward: P = J * P * J' + Q
    estimateErrorCovariance_ = (transferFunctionJacobian_ *
                                estimateErrorCovariance_ *
                                transferFunctionJacobian_.transpose());
    estimateErrorCovariance_.noalias() += delta * (*processNoiseCovariance);

    FB_DEBUG("Predicted estimate error covariance is:\n" << estimateErrorCovariance_ <<
             "\n\n--------------------- /Ekf::predict ----------------------\n");
  }

  void Ekf::predict()
  {
    FB_DEBUG("---------------------- Ekf::predict ----------------------\n" <<
             "state is " << state_ << "\n");

    double roll = state_(StateMemberRoll);
    double pitch = state_(StateMemberPitch);
    double yaw = state_(StateMemberYaw);
    double xVel = state_(StateMemberVx);
    double yVel = state_(StateMemberVy);
    double zVel = state_(StateMemberVz);
    double pitchVel = state_(StateMemberVpitch);
    double yawVel = state_(StateMemberVyaw);
    // double xAcc = state_(StateMemberAx);
    // double yAcc = state_(StateMemberAy);
    // double zAcc = state_(StateMemberAz);

    // We'll need these trig calculations a lot.
    double sp = ::sin(pitch);
    double cp = ::cos(pitch);
    double cpi = 1.0 / cp;
    double tp = sp * cpi;

    double sr = ::sin(roll);
    double cr = ::cos(roll);

    double sy = ::sin(yaw);
    double cy = ::cos(yaw);

    // compute controlAcceleration_ if use_control = true
    // prepareControl(referenceTime, delta);

    // Prepare the transfer function

    // transferFunction_(StateMemberVx, StateMemberAx) = controlDelta_;
    // transferFunction_(StateMemberVy, StateMemberAy) = controlDelta_;
    // transferFunction_(StateMemberVz, StateMemberAz) = controlDelta_;

    // Prepare the transfer function Jacobian. This function is analytically derived from the
    // transfer function.
    double xCoeff = 0.0;
    double yCoeff = 0.0;
    double zCoeff = 0.0;
    double oneHalfATSquared = 0.5 * controlDelta_ * controlDelta_;

    yCoeff = cy * sp * cr + sy * sr;
    zCoeff = -cy * sp * sr + sy * cr;
    // double dFx_dR = (yCoeff * yVel + zCoeff * zVel) * controlDelta_ +
    //                 (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    // double dFR_dR = 1.0 + (cr * tp * pitchVel - sr * tp * yawVel) * controlDelta_;

    // xCoeff = -cy * sp;
    // yCoeff = cy * cp * sr;
    // zCoeff = cy * cp * cr;
    // double dFx_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * controlDelta_ +
    //                 (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    // double dFR_dP = (cpi * cpi * sr * pitchVel + cpi * cpi * cr * yawVel) * controlDelta_;

    // yCoeff = sy * sp * cr - cy * sr;
    // zCoeff = -sy * sp * sr - cy * cr;
    // double dFy_dR = (yCoeff * yVel + zCoeff * zVel) * controlDelta_ +
    //                 (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    // double dFP_dR = (-sr * pitchVel - cr * yawVel) * controlDelta_;

    // xCoeff = -sy * sp;
    // yCoeff = sy * cp * sr;
    // zCoeff = sy * cp * cr;
    // double dFy_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * controlDelta_ +
    //                 (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;

    // yCoeff = cp * cr;
    // zCoeff = -cp * sr;
    // double dFz_dR = (yCoeff * yVel + zCoeff * zVel) * controlDelta_ +
    //                 (yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    // double dFY_dR = (cr * cpi * pitchVel - sr * cpi * yawVel) * controlDelta_;

    // xCoeff = -cp;
    // yCoeff = -sp * sr;
    // zCoeff = -sp * cr;
    // double dFz_dP = (xCoeff * xVel + yCoeff * yVel + zCoeff * zVel) * controlDelta_ +
    //                 (xCoeff * xAcc + yCoeff * yAcc + zCoeff * zAcc) * oneHalfATSquared;
    // double dFY_dP = (sr * tp * cpi * pitchVel - cr * tp * cpi * yawVel) * controlDelta_;

    // Much of the transfer function Jacobian is identical to the transfer function
    double delta_yaw = controlVelocity_(ControlMemberVyaw) * controlDelta_;
    transferFunctionJacobian_ = transferFunction_;
    transferFunctionJacobian_(StateMemberX, StateMemberYaw) = -sin(yaw+delta_yaw/2.0)*controlDelta_*controlVelocity_(ControlMemberVx);
    transferFunctionJacobian_(StateMemberY, StateMemberYaw) = cos(yaw+delta_yaw/2.0)*controlDelta_*controlVelocity_(ControlMemberVx);

    transferFunctionJacobian_(StateMemberX, StateMemberVx) = cos(yaw+delta_yaw/2.0)*controlDelta_;
    transferFunctionJacobian_(StateMemberX, StateMemberVyaw) = -sin(yaw+delta_yaw/2.0)*controlDelta_*controlDelta_/2.0;
    transferFunctionJacobian_(StateMemberY, StateMemberVx) = sin(yaw+delta_yaw/2.0)*controlDelta_;
    transferFunctionJacobian_(StateMemberY, StateMemberVyaw) = cos(yaw+delta_yaw/2.0)*controlDelta_*controlDelta_/2.0;

    // if (displayCounter_>=30)
    // {
    //   ROS_INFO_STREAM("Transfer function is:\n" << transferFunction_ <<
    //           "\nTransfer function Jacobian is:\n" << transferFunctionJacobian_ <<
    //           "\nProcess noise covariance is:\n" << processNoiseCovariance_ <<
    //           "\nCurrent state is:\n" << state_ << "\n");
    //   // std::cout << "Joc x yaw: " << -sy*controlDelta_*xVel
    //   //           << "\tJoc y yaw: " << cy*controlDelta_*xVel  << std::endl;

    //   // std::cout << "sy: " << sy
    //   //           << "\tcontrolDelta: " << controlDelta_
    //   //           << "\txVel: \n" << xVel << std::endl;
    //   displayCounter_=0;
    // }
    // displayCounter_++;

    Eigen::MatrixXd *processNoiseCovariance = &processNoiseCovariance_;

    if (useDynamicProcessNoiseCovariance_)
    {
      std::cout << "computeDynamicProcessNoiseCovariance" << std::endl;
      computeDynamicProcessNoiseCovariance(state_, controlDelta_);
      processNoiseCovariance = &dynamicProcessNoiseCovariance_;
    }

    // (1) predict x, y and yaw using control msg
    if (useControlPredict_)
    {
      state_(StateMemberX) += controlVelocity_(ControlMemberVx)* cos(yaw+delta_yaw/2.0) * controlDelta_;
      state_(StateMemberY) += controlVelocity_(ControlMemberVx) * sin(yaw+delta_yaw/2.0) * controlDelta_;
      state_(StateMemberYaw) += delta_yaw;
      // std::cout << "delta yaw/2: " << delta_yaw/2.0 << std::endl;
      // std::cout << "control vx: " << controlVelocity_(ControlMemberVx)
      //           << "\tcontrol vy: " << controlVelocity_(ControlMemberVy)
      //           << "\tcontrol vyaw: " << controlVelocity_(ControlMemberVyaw)
      //           << "\tcontrolDelta_: " << controlDelta_ <<std::endl;
    }

    // (2) Project the state forward: x = Ax + Bu (really, x = f(x, u))
    state_ = transferFunction_ * state_;

    // Handle wrapping
    wrapStateAngles();

    FB_DEBUG("Predicted state is:\n" << state_ <<
             "\nCurrent estimate error covariance is:\n" <<  estimateErrorCovariance_ << "\n");

    // (3) Project the error forward: P = J * P * J' + Q
    estimateErrorCovariance_ = (transferFunctionJacobian_ *
                                estimateErrorCovariance_ *
                                transferFunctionJacobian_.transpose());
    estimateErrorCovariance_.noalias() += ((transferFunctionJacobian_*
                                          (*processNoiseCovariance) *
                                          transferFunctionJacobian_.transpose()) +
                                          0.1*(*processNoiseCovariance)) * controlDelta_; // + (J*Qu*J' + Q1)

    FB_DEBUG("Predicted estimate error covariance is:\n" << estimateErrorCovariance_ <<
             "\n\n--------------------- /Ekf::predict ----------------------\n");

  }

}  // namespace RobotLocalization
