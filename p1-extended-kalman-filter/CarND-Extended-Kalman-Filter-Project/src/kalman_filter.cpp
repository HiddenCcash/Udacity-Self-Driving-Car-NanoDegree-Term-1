#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define PI 3.14159265

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in; // includes noise from measurment
  Q_ = Q_in; // includes noise from acceleration
}

void KalmanFilter::Predict() {

  //predict calculations
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

  //update calcuations
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  //update calculations
  //recover state parameters
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  //check division by zero
  if(fabs(px) < 0.0001){
    px = 0.0001;
  }
  float phi = atan2(py, px);
  // convert cartesian coords. to polar and add to z_pred.
  Eigen::VectorXd z_pred = VectorXd(3);
  z_pred << sqrt(pow(px,2)+pow(py,2)),
            phi,
            (px*vx + py*vy)/sqrt(pow(px,2)+pow(py,2));
  VectorXd y = z - z_pred;
  // NOTE!
  // We need to make sure to "normalize" phi in the y vector (which is `y(1)`)
  // so that its angle is between -pi and pi;
  // in other words, add or subtract 2pi from phi until it is between -pi and pi.
  y(1) = fmod(y(1), 2*PI);

  //continue on with Meas. Update
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}
