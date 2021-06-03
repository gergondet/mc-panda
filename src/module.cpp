#include "panda.h"

#include <mc_rbdyn/RobotModuleMacros.h>
#include <mc_rtc/logging.h>

extern "C"
{
  ROBOT_MODULE_API void MC_RTC_ROBOT_MODULE(std::vector<std::string> & names)
  {
    names = {"Panda", "PandaDefault", "PandaHand", "PandaPump", "PandaFoot"};
  }
  ROBOT_MODULE_API void destroy(mc_rbdyn::RobotModule * ptr)
  {
    delete ptr;
  }
  ROBOT_MODULE_API mc_rbdyn::RobotModule * create(const std::string & n)
  {
    ROBOT_MODULE_CHECK_VERSION("Panda")
    if(n == "Panda" || n == "PandaDefault")
    {
      return new mc_robots::PandaRobotModule(false, false, false);
    }
    else if(n == "PandaPump")
    {
      return new mc_robots::PandaRobotModule(true, false, false);
    }
    else if(n == "PandaFoot")
    {
      return new mc_robots::PandaRobotModule(false, true, false);
    }
    else if(n == "PandaHand")
    {
      return new mc_robots::PandaRobotModule(false, false, true);
    }
    else
    {
      mc_rtc::log::error("Panda module cannot create an object of type {}", n);
      return nullptr;
    }
  }
}
