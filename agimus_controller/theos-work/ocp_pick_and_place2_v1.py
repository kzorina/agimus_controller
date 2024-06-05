"""
@package ddp_mpc
@file ocp_pick_and_place2.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2019, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Simple self-contained example of DDP trajectory for KUKA - without PyBullet 
"""

import numpy as np
import pinocchio as pin
from mim_robots.robot_loader import load_pinocchio_wrapper
import crocoddyl
import mim_solvers
import example_robot_data

###################
### ROBOT MODEL ###
###################
# pin_robot = load_pinocchio_wrapper("iiwa")
pin_robot = load_pinocchio_wrapper("panda")
locked_joints = [
    pin_robot.model.getJointId("panda_finger_joint1"),
    pin_robot.model.getJointId("panda_finger_joint2"),
]
robot_model_reduced = pin.buildReducedModel(
    pin_robot.model, locked_joints, pin_robot.q0
)
pin_robot.model = robot_model_reduced

# id_endeff = pin_robot.model.getFrameId("contact") # KUKA
id_endeff = pin_robot.model.getFrameId("panda_joint7")  # PANDA
nq = pin_robot.model.nq
nv = pin_robot.model.nv
# Hard-coded initial state of StartDGM application in KUKA sunrise control panel
q0 = np.array([0.0, 0.349066, 0.0, -0.872665, 0.0, 0.0, 0.0])

### CHECKOUT I changed the initial config
q0 = np.array([1.5, 0.5, 0.0, -1.8, 0.0, 0.0, 0.0])  # KUKA
q0 = pin_robot.q0[:7]  # PANDA

dq0 = pin.utils.zero(nv)
# Update pinocchio model with forward kinematics and get frame initial placemnent + desired frame placement
pin.forwardKinematics(pin_robot.model, pin_robot.data, q0)
pin.updateFramePlacements(pin_robot.model, pin_robot.data)
M0 = pin_robot.data.oMf[id_endeff]

##########
# VIEWER #
##########
import time


pin_robot.initViewer(loadModel=True)


def disp(xs):
    for x in xs:
        pin_robot.display(x[:7])
        time.sleep(0.005)


pin_robot.viz.viewer.gui.addSphere("world/target1", 0.05, [1, 0, 0, 1])
pin_robot.viewer.gui.applyConfiguration("world/target1", [-0.5, 0.5, 0.2, 0, 0, 0, 1])
pin_robot.viz.viewer.gui.addSphere("world/target2", 0.05, [1, 0, 0, 1])
pin_robot.viewer.gui.applyConfiguration("world/target2", [0.5, 0.5, 0.2, 0, 0, 0, 1])
pin_robot.viewer.gui.refresh()
# pin_robot.viz.viewer.gui.addSphere('world/target1',.1,[1,0,0,1])
# pin_robot.viz.viewer.gui.addSphere('world/target2',.1,[0,2,0,1])


#################
### OCP SETUP ###
#################
# Set up the OCP for Crocoddyl define (cost models, shooting problem and solver)
#     x0                     : initial state
#     desiredFramePlacement  : desired end-effector placement (pin.SE3)
#     runningCostWeights     : running cost function weights  [frame_placement, state_reg, ctrl_reg, state_limits]
#     terminalCostWeights    : terminal cost function weights [frame_placement, state_reg, state_limits]
#     stateWeights           : activation weights for state regularization in running cost [w_x1,...,w_xn]
#     stateWeightsTerm       : activation weights for state regularization in terminal cost [w_x1,...,w_xn]
#     framePlacementWeights  : activation weights for frame placement in running & terminal cost [w_px, w_py, w_pz, w_Rx, w_Ry, w_Rz]
# The running cost looks like this   : l(x,u) = ||log_SE3|| + ||x|| + ||u|| + QuadBarrier(x)
# The terminal cost looks like this  : l(x)   = ||log_SE3|| + ||x|| + QuadBarrier(x)
### OCP param + initialization
# Integration step for DDP (s)
DT = 5e-2  # 2e-2
# Number of knots in the cycle
NCYCLE = 40  # 80 # 40
# Number of knots in the MPC horizon
NHORIZON = int(NCYCLE / 2.0)  # NCYCLE *2//3  # 40
# Initialize OCP
x0 = np.concatenate([q0, dq0])
# Initial guess = duplicate initial state and gravity compensation torque
xs = [x0] * (NHORIZON + 1)
us = [
    pin.rnea(
        pin_robot.model, pin_robot.data, x0[:nq], np.zeros((nv, 1)), np.zeros((nq, 1))
    )
] * NHORIZON

# Cost weights and reference
# Desired frame placement + cost weights
desiredFramePlacement = pin.SE3(np.eye(3), np.array([-0.5, -0.5, 0.2]))
runningCostWeights = [1, 1e-1, 1e-2, 30.0]  # EE placement , xreg , ureg , xlim
terminalCostWeights = [10, 1e-1, 30.0]  # EE placement , xreg , xlim
stateWeights = np.array([1.0] * nq + [1.0] * nv)  # size nq + nv
stateWeightsTerm = np.array([1.0] * nq + [1.0] * nv)  # size nq + nv
framePlacementWeights = np.array([10.0] * 3 + [0.0] * 3)  # size 6 (3 pos + 3 rot)
# Desired frame velocity + cost weights
desiredFrameMotion = pin.Motion(
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
)  # size 6 (3 lin , 3 ang)
frameVelocityRunningCostWeight = 0.0
frameVelocityTerminalCostWeight = 0.0
frameVelocityWeights = np.array(
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
)  # size 6 (3 lin , 3 ang)

# Construct OCP
# State and actuation models
state = crocoddyl.StateMultibody(pin_robot.model)
actuation = crocoddyl.ActuationModelFull(state)
# State  & regularizations
xRegCost = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
    crocoddyl.ResidualModelState(state, x0, actuation.nu),
)
stateWeightsTerm[:7] = 0.1
xRegCostTerm = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ActivationModelWeightedQuad(stateWeightsTerm**2),
    crocoddyl.ResidualModelState(state, x0, actuation.nu),
)

ugrav = pin.computeGeneralizedGravity(pin_robot.model, pin_robot.data, q0)[:7].copy()
"""
ugrav2 = np.array(
    [0.0, -54.87, -1.147, 11.006, 0.674, -0.752, 0.019]
)  ## hardcoded from terminal config ... beurgh"""
change = 0.4
# ugrav = ugrav * (1 - change) + ugrav2 * change

uRegCost = crocoddyl.CostModelResidual(
    state, crocoddyl.ResidualModelControl(state, uref=ugrav)
)
# Adding the state limits penalization
xLimitCost = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(state.lb, state.ub)
    ),
    crocoddyl.ResidualModelState(state, 0 * x0, actuation.nu),
)
# Running and terminal cost model


def createModel(des, wpos, wvel, wregx, wregu, wlimx):
    runningCostModel = crocoddyl.CostModelSum(state, nu=actuation.nu)
    # framePlacement = crocoddyl.FramePlacement(id_endeff, desiredFramePlacement)
    # framePlacementCost = crocoddyl.CostModelFramePlacement(state,
    #                                                        framePlacement, actuation.nu)
    framePosition = des
    framePlacementCost = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ResidualModelFrameTranslation(
            state, id_endeff, framePosition, actuation.nu
        ),
    )
    frameVel = desiredFrameMotion
    frameVelocityCost = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ResidualModelFrameVelocity(state, id_endeff, frameVel, pin.WORLD),
    )

    # Add up running cost terms
    runningCostModel.addCost("endeff", framePlacementCost, wpos)
    runningCostModel.addCost("stateReg", xRegCost, wregx)
    runningCostModel.addCost("ctrlReg", uRegCost, wregu)
    runningCostModel.addCost("stateLim", xLimitCost, wlimx)
    runningCostModel.addCost("endeff_vel", frameVelocityCost, wvel)

    # Create IAMs
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        ),
        DT,
    )
    runningModel.differential.armature = np.array([0.1] * 7)
    return runningModel


target1 = np.array([-0.5, 0.5, 0.2])
target2 = np.array([0.5, 0.5, 0.2])

# target1 = np.array([-0.3, 0.5, 0.2])
# target2 = np.array([ 0.3, 0.5, 0.2])

# pin_robot.viz.viewer.gui.applyConfiguration('world/target1', target1.tolist()+[0,0,0,1])
# pin_robot.viz.viewer.gui.applyConfiguration('world/target2', target2.tolist()+[0,0,0,1])
# pin_robot.viz.viewer.gui.applyConfiguration('world/target2', target2.tolist()+[0,0,0,1])
# pin_robot.viz.viewer.gui.refresh()

wregx = 0.1
wregu = 0.08
# Create the shooting problem
runningModels = [
    createModel(target1, wpos=0.1, wvel=0, wregx=wregx, wregu=wregu, wlimx=50)
    for i in range(NHORIZON)
]
terminalModel = createModel(target2, wpos=0, wvel=0, wregx=0.1, wregu=0, wlimx=0)


def resetProblem(t, x, problem, check=True):
    """
    Given the current time <t> in milliseconds, change the costs of the list of models <models>
    to follow the refenrence parameters describing the robot cycle.
    We assume that the model length covers half a cycle.
    """
    models = list(problem.runningModels)
    problem.x0 = x

    TCYCLE = DT * NCYCLE
    TCYCLE_2 = TCYCLE / 2

    t0cycle = t // TCYCLE * TCYCLE  # Date of cycle start in ms
    t0cycleB = (
        t0cycle + TCYCLE_2 if t > t0cycle + TCYCLE_2 else t0cycle - TCYCLE_2
    )  # Date of start of the anti cycle
    # print('t0 = ',t0cycle, "\tt = ",t)

    WEIGHT_SLOPE = 18  # 20 10
    WEIGHT_CUT = 0.79  # 0.8 # 0.5 * TCYCLE

    # print("t0-cycle = ", t0cycle)
    ### Change the models according to the initial time
    # The weights are computed from the integration (over each shooting interval) of
    # a given periodic weight function.
    # First trial: - weights(t) = exp(slope*(t-tcut))
    current_target = None
    for k, m in enumerate(models[:]):

        # Compute the absolute time of the shooting interval, modulo the cycle time,
        # so that 0<=ta0<TCYCLE and 0<ta1<=TCYCLE

        ta0 = (
            t + k * DT - t0cycle
        )  # absolute data of the time of the start of the shooting interval
        ta1 = (
            t + (k + 1) * DT - t0cycle
        )  # absolute data of the time of the start of the shooting interval
        if ta0 > TCYCLE:
            ta0 -= TCYCLE
            ta1 -= TCYCLE
        # print(ta0,ta1)

        # Compute the absolute time of the shooting interval for the second task, modulo the cycle time,
        # so that 0<=tb0<TCYCLE and 0<tb1<=TCYCLE and [ta0,ta1] is in antiphase with [tb0,tb1].
        if ta0 < TCYCLE_2:
            tb0 = ta0 + TCYCLE_2
            tb1 = ta1 + TCYCLE_2 if ta1 <= TCYCLE_2 else TCYCLE
        else:
            tb0 = ta0 - TCYCLE_2
            tb1 = ta1 - TCYCLE_2
        if ta1 > TCYCLE:
            ta1 = TCYCLE
        # print(tb0,tb1)
        assert 0 <= ta0 <= ta1 <= TCYCLE
        assert 0 <= tb0 <= tb1 <= TCYCLE

        # Compute the weights for the first task, as \integral_ta0^ta1 weight(s) ds.
        e1 = np.exp(WEIGHT_SLOPE * (ta1 - WEIGHT_CUT))
        e0 = np.exp(WEIGHT_SLOPE * (ta0 - WEIGHT_CUT))
        weight_a = (e1 - e0) / WEIGHT_SLOPE
        assert weight_a < (e1 + e0) / 2 * DT

        # Compute the weights for the second task, as \integral_tb0^tb1 weight(s) ds.
        e1 = np.exp(WEIGHT_SLOPE * (tb1 - WEIGHT_CUT))
        e0 = np.exp(WEIGHT_SLOPE * (tb0 - WEIGHT_CUT))
        weight_b = (e1 - e0) / WEIGHT_SLOPE

        # The cost would be wa*(p-pa)^2 + wb*(p-pb)^2 = (wa+wb)*(p - (wa.pa+wb.pb)/(wa+wb).
        # Set w=wa+wb, and target=weightedsum((pa,wa),(pb,wb)).
        weight = weight_a + weight_b
        # print(weight)
        target = (weight_a * target1 + weight_b * target2) / weight
        if k == 0:
            current_target = target
        costs = m.differential.costs
        # costs.costs["endeff"].cost.reference = target
        # breakpoint()
        costs.costs["endeff"].cost.residual.reference = target
        costs.costs["endeff"].weight = weight
        costs.costs["endeff_vel"].weight = weight / 50
        m.dt = DT
        """
        print(
            "%d:\t  [%.2f,%.2f] = %.2f \t [%.2f,%.2f] = %.2f \t T=%.3f"
            % (k, ta0, ta1, weight_a, tb0, tb1, weight_b, target[0])
        )"""

        assert 0 <= weight_a and 0 <= weight_b and weight > 0

    # Check (optionnal)
    if not check:
        return
    return current_target
    """
    # stophere
    import matplotlib.pylab as plt

    
    
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot([m.differential.costs.costs["endeff"].weight for m in models])
    # print(sum([ m.differential.costs.costs['endeff'].weight for m in models]))
    plt.axis([-5, 45, -5, 1200])
    plt.subplot(212)
    # print(m.differential.costs.costs['endeff'].cost.reference)
    plt.plot([m.differential.costs.costs["endeff"].cost.reference[0] for m in models])
    plt.draw()
    # time.sleep(1e-3)
    plt.pause(1e-3)
    """


# Create the shooting problem
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
# Creating the DDP solver
solver = mim_solvers.SolverSQP(problem)

# SOLVE
# solver.solve([],[],maxiter=1000)
# xs = np.array(solver.xs)
# us = np.array(solver.us)

solver.th_stop = 1e-9
hx = []
hu = []
hp = []
hiter = []
targets = []
x = x0.copy()

for n in terminalModel.differential.costs.costs.todict().keys():
    terminalModel.differential.costs.costs[n].weight = 0

NSTEPS = 1  # Recompute the MPC every NSTEPS
NTOTAL = 300
for t in range(0, NTOTAL, NSTEPS):
    if not t % 100:
        print("Simu time %d" % t)
    target = resetProblem(t * 1e-3, x, solver.problem)  #
    # targets.append(target)
    solver.solve(solver.xs, solver.us, 10)
    # print("TORQUE = ", solver.us[0])
    hx.append(solver.xs[0].copy())
    hu.append(solver.us[0].copy())

    hp.append(
        solver.problem.runningDatas[0]
        .differential.multibody.pinocchio.oMf[id_endeff]
        .translation.copy()
    )
    hiter.append(solver.iter)

    # Compute the next state by integrating using the first shooting model (hackee)
    m = solver.problem.runningModels[0]
    m.dt = 1e-3 * NSTEPS
    d = m.createData()
    m.calc(d, x, solver.us[0])
    x = d.xnext.copy()


# # ### DISPLAY AND PLOT
# # disp(hx,1e-3*NSTEPS)

import matplotlib.pyplot as plt  # ; plt.ion()

"""
for i in range(nq):
    plt.subplot(nq, 1, i + 1)
    plt.plot([x[i] for x in hu], "-")
plt.show()
"""
axis_string = ["x", "y", "z"]
for i in range(nq):
    plt.subplot(nq, 1, i + 1)
    plt.plot([x[i] for x in hu], "-")
    plt.ylabel(f"q{i} torque")
plt.show()

for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot([x[i] for x in hp], "-")
    plt.plot([target[i] for target in targets], "-")
    plt.ylabel("end effector " + axis_string[i] + " position")
    plt.legend(["mpc", "target"], loc="best")
plt.show()


# # Dump in data file for dg_demos impedance_offline.py
# # Reshape trajs
# xs = np.array(hx)
# qs = xs[:,:nq]
# vs = xs[:,nv:]
# # Add constant desired state at the end of the trajectory to avoid reader bug (loops back to start)
# N_end = 10000
# qs_tot = np.vstack([qs, np.array([qs[-1,:]]*N_end)])
# vs_tot = np.vstack([vs, np.array([vs[-1,:]]*N_end)])
# # Same for controls, using gravity compensation
# qs_end = qs_tot[-1,:]
# u_grav = pin.rnea(pin_robot.model, pin_robot.data, qs_end, np.zeros((nv,1)), np.zeros((nq,1)))
# us_tot = np.vstack([np.array(hu), np.array([u_grav]*N_end)])
# # Dump to data files
# np.savetxt("/tmp/iiwa_ddp_pos_traj.dat", qs_tot, delimiter=" ")
# np.savetxt("/tmp/iiwa_ddp_vel_traj.dat", vs_tot, delimiter=" ")
# np.savetxt("/tmp/iiwa_ddp_tau_traj.dat", us_tot, delimiter=" ")
