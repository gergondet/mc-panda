#pragma once

#include <mc_rbdyn/RobotModule.h>

#include <mc_panda/api.h>

namespace mc_robots
{

struct MC_PANDA_DLLAPI PandaRobotModule : public mc_rbdyn::RobotModule
{
public:
  PandaRobotModule(bool pump, bool foot, bool hand);
};

} // namespace mc_robots
