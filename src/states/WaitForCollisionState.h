#pragma once

#include <mc_control/CompletionCriteria.h>
#include <mc_control/fsm/State.h>
#include "../devices/PandaSensor.h"
#include "../controller/PandaController.h"

struct WaitForCollisionState : mc_control::fsm::State
{
    void configure(const mc_rtc::Configuration & config) override;

    void start(mc_control::fsm::Controller & ctl) override;

    bool run(mc_control::fsm::Controller & ctl) override;

    void teardown(mc_control::fsm::Controller & ctl) override;

  private:
    mc_rtc::Configuration state_conf_;

    bool sensorAvailable = false;
    std::shared_ptr<mc_panda::PandaSensor> sensor;
    std::string sensorDeviceName = "PandaSensor";

    bool collisionDetected = false;
    Eigen::Matrix<double, 7, 1> joint_contactVector_;
    Eigen::VectorXd joint_contactVector_thresholds_;
    Eigen::Matrix<double, 6, 1> cartesian_contactVector_;
    Eigen::VectorXd cartesian_contactVector_thresholds_;
    double forceThreshold_ = 10;
    mc_rbdyn::ForceSensor forceSensor;
};