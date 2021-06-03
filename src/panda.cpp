#include "panda.h"
#include "devices/Pump.h"
#include "devices/Robot.h"

#include "config.h"

#ifndef M_PI
#  include <boost/math/constants/constants.hpp>
#  define M_PI boost::math::constants::pi<double>()
#endif

#include <mc_rbdyn/rpy_utils.h>
#include <mc_rtc/constants.h>

#include <RBDyn/parsers/urdf.h>

namespace mc_robots
{

inline static std::string pandaVariant(bool pump, bool foot, bool hand)
{
  if(pump && !foot && !hand)
  {
    mc_rtc::log::info("PandaRobotModule uses the panda variant: 'panda_pump'");
    return "panda_pump";
  }
  if(!pump && foot && !hand)
  {
    mc_rtc::log::info("PandaRobotModule uses the panda variant: 'panda_foot'");
    return "panda_foot";
  }
  if(!pump && !foot && hand)
  {
    mc_rtc::log::info("PandaRobotModule uses the panda variant: 'panda_hand'");
    return "panda_hand";
  }
  if(!pump && !foot && !hand)
  {
    mc_rtc::log::info("PandaRobotModule uses the panda variant: 'panda_default'");
    return "panda_default";
  }
  mc_rtc::log::error("PandaRobotModule does not provide this panda variant...");
  return "";
}

PandaRobotModule::PandaRobotModule(bool pump, bool foot, bool hand)
: RobotModule(PANDA_DESCRIPTION_PATH, pandaVariant(pump, foot, hand))
{
  mc_rtc::log::success("PandaRobotModule loaded with name: {}", name);
  urdf_path = path + "/" + name + ".urdf";
  _real_urdf = urdf_path;
  init(rbd::parsers::from_urdf_file(urdf_path, true));

  // additional joint panda limits, see https://frankaemika.github.io/docs/control_parameters.html#constants
  using bound_t = mc_rbdyn::RobotModule::accelerationBounds_t::value_type;
  bound_t torqueDerivativeUpper;
  bound_t torqueDerivativeLower;
  bound_t accelerationBoundsUpper;
  bound_t accelerationBoundsLower;
  for(const auto & b : _bounds[0])
  {
    torqueDerivativeUpper[b.first] = std::vector<double>(b.second.size(), 1000);
    torqueDerivativeLower[b.first] = std::vector<double>(b.second.size(), -1000);
  }
  accelerationBoundsLower = {{"panda_joint1", {-15}},   {"panda_joint2", {-7.5}}, {"panda_joint3", {-10}},
                             {"panda_joint4", {-12.5}}, {"panda_joint5", {-15}},  {"panda_joint6", {-20}},
                             {"panda_joint7", {-20}}};
  accelerationBoundsUpper = {{"panda_joint1", {15}},   {"panda_joint2", {7.5}}, {"panda_joint3", {10}},
                             {"panda_joint4", {12.5}}, {"panda_joint5", {15}},  {"panda_joint6", {20}},
                             {"panda_joint7", {20}}};
  _torqueDerivativeBounds.push_back(torqueDerivativeLower);
  _torqueDerivativeBounds.push_back(torqueDerivativeUpper);
  _accelerationBounds.push_back(accelerationBoundsLower);
  _accelerationBounds.push_back(accelerationBoundsUpper);

  rsdf_dir = path + "/rsdf/" + name + "/";
  calib_dir = path + "/calib";

  _bodySensors.clear();

  _stance["panda_joint1"] = {mc_rtc::constants::toRad(0)};
  _stance["panda_joint2"] = {mc_rtc::constants::toRad(0)};
  _stance["panda_joint3"] = {mc_rtc::constants::toRad(0)};
  _stance["panda_joint4"] = {mc_rtc::constants::toRad(-120)};
  _stance["panda_joint5"] = {mc_rtc::constants::toRad(0)};
  _stance["panda_joint6"] = {mc_rtc::constants::toRad(120)};
  _stance["panda_joint7"] = {mc_rtc::constants::toRad(0)};

  _forceSensors.push_back(
      mc_rbdyn::ForceSensor("LeftHandForceSensor", "panda_link7",
                            sva::PTransformd(mc_rbdyn::rpyToMat(3.14, 0.0, 0.0), Eigen::Vector3d(0, 0, -0.04435))));

  if(foot)
  {
    _convexHull["panda_foot"] = {"panda_foot", path + "/convex/panda_foot/panda_foot-ch.txt"};
  }
  if(pump)
  {
    _convexHull["panda_pump"] = {"panda_pump", path + "/convex/panda_pump/panda_pump-ch.txt"};
  }

  const double i = 0.015; // 0.01;
  const double s = 0.0075; // 0.005;
  const double d = 0.;
  _minimalSelfCollisions = {{"panda_link0*", "panda_link5*", i, s, d},
                            {"panda_link1*", "panda_link5*", i, s, d},
                            {"panda_link2*", "panda_link5*", i, s, d},
                            {"panda_link3*", "panda_link5*", i, s, d},
                            {"panda_link0*", "panda_link6*", i, s, d},
                            {"panda_link1*", "panda_link6*", i, s, d},
                            {"panda_link2*", "panda_link6*", i, s, d},
                            {"panda_link3*", "panda_link6*", i, s, d},
                            {"panda_link0*", "panda_link7*", i, s, d},
                            {"panda_link1*", "panda_link7*", i, s, d},
                            {"panda_link2*", "panda_link7*", i, s, d},
                            {"panda_link3*", "panda_link7*", i, s, d},
                            // FIXME Is this last one needed?
                            {"panda_link5*", "panda_link7*", i, s, d}};

  /* Additional self collisions */
  if(pump)
  {
    // FIXME No pump convex ATM
    //_commonSelfCollisions.push_back({"panda_link0", "pump", i, s, d)};
    //_commonSelfCollisions.push_back({"panda_link1", "pump", i, s, d)};
    //_commonSelfCollisions.push_back({"panda_link2", "pump", i, s, d)};
    //_commonSelfCollisions.push_back({"panda_link3", "pump", i, s, d)};
  }
  if(foot)
  {
    _commonSelfCollisions.push_back({"panda_link0", "foot", i, s, d});
    _commonSelfCollisions.push_back({"panda_link1", "foot", i, s, d});
    _commonSelfCollisions.push_back({"panda_link2", "foot", i, s, d});
    _commonSelfCollisions.push_back({"panda_link3", "foot", i, s, d});
  }
  if(hand)
  {
    _commonSelfCollisions.push_back({"panda_link0", "hand", i, s, d});
    _commonSelfCollisions.push_back({"panda_link1", "hand", i, s, d});
    _commonSelfCollisions.push_back({"panda_link2", "hand", i, s, d});
    _commonSelfCollisions.push_back({"panda_link3", "hand", i, s, d});
  }

  _commonSelfCollisions = _minimalSelfCollisions;

  _ref_joint_order = {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                      "panda_joint5", "panda_joint6", "panda_joint7"};

  _default_attitude = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  // NOTE: this is a joint-space sensor and not attached to a specific Cartesian link with a certain transformation
  _devices.emplace_back(new mc_panda::Robot());
  if(pump)
  {
    /* Pump device */
    _devices.emplace_back(new mc_panda::Pump("panda_link8", sva::PTransformd::Identity()));
  }

  /* Grippers */
  if(hand)
  {
    // Module wide gripper configuration
    _gripperSafety = {0.15, 1.0};
    _grippers = {{"gripper", {"panda_finger_joint1"}, false}};
    _ref_joint_order.push_back("panda_finger_joint1");
    _ref_joint_order.push_back("panda_finger_joint2");
  }

  mc_rtc::log::success("PandaRobotModule uses urdf_path {}", urdf_path);
  mc_rtc::log::success("PandaRobotModule uses rsdf_dir {}", rsdf_dir);
}

} // namespace mc_robots
