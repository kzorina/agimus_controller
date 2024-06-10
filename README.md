# agimus_controller

## Getting started

In order to build this package one need various complexe codes:

- Humanoid Path Planneur
- Agimus software
- Croccodyl

All of these are built in a single docker:
gitlab.laas.fr:4567/agimus-project/agimus_dev_container:noetic-devel

One can simply use this package in order to use the docker in the VSCode
development editor.
https://gitlab.laas.fr/agimus-project/agimus_dev_container

## Run one application.

### Pick an place scenario using HPP

We use the Humnaoid Path Planner in order to get a path and then build a whole-body model predictive controller that tracks the planned trajectory.

Once the code is built one can run these in several terminals inside the docker:
- `roscore`
- `hppcorbaserver`
- `gepetto-gui`
- run the script:
    - with python3: `python3 -i -m agimus_controller.main_hpp_mpc -N=1`
    - with ipython3: `ipython3 -i -m agimus_controller.main_hpp_mpc -- -N=1`

For one to simply run the node individually.

```bash
rosrun agimus_controller agimus_controller_node
```

For a more complete setup see the
https://github.com/agimus-project/agimus_pick_and_place
package.
