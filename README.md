mc_panda
========

[![License](https://img.shields.io/badge/License-BSD%202--Clause-green.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![CI](https://github.com/jrl-umi3218/mc_panda/workflows/CI%20of%20mc_panda/badge.svg?branch=master)](https://github.com/jrl-umi3218/mc_panda/actions?query=workflow%3A%22CI+of+mc_panda%22)

This package provides the following robot modules for [mc_rtc]:
- `PandaDefault`: panda robot without an end-effector mounted
- `PandaFoot`: panda robot with a static foot-like end-effector
- `PandaHand`: panda robot with the standard panda gripper
- `PandaPump`: panda robot with the standard panda pump

It also contains the definition of two devices:
- `mc_panda::Robot` is an asynchronous interface for `franka::Robot` commands and `franka::RobotState` which are connected to the actual robot when [mc_franka] is running the controller;
- `mc_panda::Pump` is an asynchronous interface for `franka::VacuumGripper` which is connected to the pump when [mc_franka] is running the controller. It is only available in the `PandaPump` variant.

And it provides some panda-specific FSM states:
- `PandaStop`: immediately stop panda motion execution
- `PandaWaitForCollision`: monitor panda contact sensor until certain threshold are reached
- `PumpDropOff`: sends a dropOff command to the panda's pump and wait for the command completion
- `PumpStop`: interrupt the pump operation
- `PumpVacuum`: sends a vacuum command to the panda's pump and wait for the command completion

See [state examples](src/states/examples.yaml) for details on the available parameters for each state.

Dependencies
------------

This package requires:
- [mc_rtc]
- [libfranka](https://github.com/frankaemika/libfranka)
- [franka_ros]

If [mc_openrtm](https://github.com/jrl-umi3218/mc_openrtm) is installed this will also install compatible Choreonoid projects for the panda robot.

Original xacro files from:
- [franka_ros] for the main definitions and collisions specifications
- [franka_gazebo](https://github.com/mkrizmancic/franka_gazebo) for the inertial parameters

Pump model was downloaded from [Schmalz](https://www.schmalz.com/en/10.03.01.00314)


### Video presentation

A video demonstrating panda motion generation and simultaneous pump actuation employing this implementation is available here:

[![Video presentation](https://img.youtube.com/vi/juynq6x9JJ8/0.jpg)](https://youtu.be/juynq6x9JJ8 "Safe Impacts with Soft Contacts Based on Learned Deformations")

### Reference

Writing code takes time.
If this implementation is useful for your research, please cite the related publication:

```
@unpublished{dehio2020arxiv,
  title = {Safe Impacts with Soft Contacts Based on Learned Deformations},
  author = {Dehio, Niels and Kheddar, Abderrahmane},
  pdf = {https://hal.archives-ouvertes.fr/hal-02973947/document},
  url = {https://hal.archives-ouvertes.fr/hal-02973947},
  month = {October},
  year = {2020}
}
```

[![I.AM.Logo](https://i-am-project.eu/templates/yootheme/cache/iam_logo-horizontaal_XL-9e4a8a2a.png)](https://i-am-project.eu/index.php)

This work was partially supported by the Research Project I.AM. through the European Union H2020 program under GA 871899.

[mc_rtc]: https://github.com/jrl-umi3218/mc_rtc
[mc_franka]: https://github.com/jrl-umi3218/mc_franka
[franka_ros]: https://github.com/frankaemika/franka_ros
