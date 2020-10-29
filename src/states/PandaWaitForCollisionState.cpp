#include "PandaWaitForCollisionState.h"

#include <mc_tasks/MetaTaskLoader.h>

#include <devices/Robot.h>

#include <spdlog/fmt/bundled/ranges.h>

namespace mc_panda
{

void PandaWaitForCollisionState::configure(const mc_rtc::Configuration & config)
{
  config("robot", robot_);
  config("jointContactThresholds", joint_contact_thresholds_);
  config("cartesianContactThresholds", cartesian_contact_thresholds_);
  config("pressureThreshold", pressure_threshold_);
}

void PandaWaitForCollisionState::start(mc_control::fsm::Controller & ctl_)
{
  if(robot_.empty())
  {
    robot_ = ctl_.robot().name();
  }
  device_ = Robot::get(ctl_.robot(robot_));
  if(!device_)
  {
    mc_rtc::log::warning("[{}] State started with robot {} which does not have an mc_panda::Robot device", name(),
                         robot_);
    output("NotAPanda");
    return;
  }
  addToLogger(ctl_.logger());
}

bool PandaWaitForCollisionState::run(mc_control::fsm::Controller & ctl_)
{
  if(output().size())
  {
    return true;
  }

  const auto & state = device_->state();
  const auto & robot = ctl_.robot(robot_);
  const auto & sensor = robot.forceSensors()[0];
  for(size_t i = 0; i < joint_contact_thresholds_.size(); ++i)
  {
    if(fabs(state.tau_ext_hat_filtered[i]) > joint_contact_thresholds_[i])
    {
      mc_rtc::log::info("[{}] Detected a joint-space collision for joint {}, value: {}, threshold: {}", name(),
                        robot.module().ref_joint_order()[i], state.tau_ext_hat_filtered[i],
                        joint_contact_thresholds_[i]);
      output("OK");
      return true;
    }
  }
  for(size_t i = 0; i < cartesian_contact_thresholds_.size(); ++i)
  {
    if(fabs(state.K_F_ext_hat_K[i]) > cartesian_contact_thresholds_[i])
    {
      std::array<char, 6> idxToCartesian{'x', 'y', 'z', 'R', 'P', 'Y'};
      mc_rtc::log::info("[{}] Detected a cartesian-space collision for axis {}, value: {}, threshold: {}", name(),
                        idxToCartesian[i], state.K_F_ext_hat_K[i], cartesian_contact_thresholds_[i]);
      output("OK");
      return true;
    }
  }
  if(fabs(sensor.force().z()) > pressure_threshold_)
  {
    mc_rtc::log::info("[{}] Detected a pressure contact, value: {}, threshold: {}", name(), sensor.force().z(),
                      pressure_threshold_);
    output("OK");
    return true;
  }
  return false;
}

void PandaWaitForCollisionState::teardown(mc_control::fsm::Controller & ctl_)
{
  removeFromLogger(ctl_.logger());
}

void PandaWaitForCollisionState::addToLogger(mc_rtc::Logger & logger)
{
  logger.addLogEntry(name() + "_joint_contact_thresholds",
                     [this]() -> const std::array<double, 7> & { return joint_contact_thresholds_; });
  logger.addLogEntry(name() + "_cartesian_contact_thresholds",
                     [this]() -> const std::array<double, 6> & { return cartesian_contact_thresholds_; });
}

void PandaWaitForCollisionState::removeFromLogger(mc_rtc::Logger & logger)
{
  logger.removeLogEntry(name() + "_joint_contact_thresholds");
  logger.removeLogEntry(name() + "_cartesian_contact_thresholds");
}

} // namespace mc_panda

EXPORT_SINGLE_STATE("PandaWaitForCollision", mc_panda::PandaWaitForCollisionState)
