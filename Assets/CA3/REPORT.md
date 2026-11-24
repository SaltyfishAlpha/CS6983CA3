# CA3 Report

**Name:**  Jinghao Zheng
**Unity version:**  6000.3.17f1
**MuJoCo plugin version:**  3.17.0
**Chosen arm (Menagerie):**  KUKA LBR iiwa 14
**Gallery / repo link:**  https://github.com/SaltyfishAlpha/CS6983CA3

## Why this arm
The iiwa 14 is a single 7-DOF arm with position-servo actuators, 
so STATE = 14 and ACTION = 7 match the skeleton directly and ctrl 
is simply a target joint angle. It has no gripper, which keeps the focus on arm dynamics, 
and its redundancy makes the IK problem interesting.

## Pipeline summary
ArmJointController samples a random joint goal near home every 3s 
and rate-limits the command to 0.05 rad per physics step; RolloutRecorder 
logs 70,000 transitions (q, qd, q*, q', qd') at 0.02 s per step. train.py 
fits an MLP that predicts the normalized state delta, using a one-step plus 
10-step rollout loss, and exports dynamics.onnx for Sentis. train_controller.py 
trains a 105-parameter linear policy with CMA-ES entirely inside the world model, 
and WorldModelController runs the same policy and smoothing in Unity. At deployment, 
IkGoalProvider turns a draggable target sphere into a joint goal with damped least squares,
and the learned controller tracks that goal.

## World model results
- DOF (`ACTION`): 7
- One-step joint RMSE: 0.0050
- Open-loop 10 / 50 / 100-step RMSE: 0.035 / 0.047 / 0.049

## Controller results
- Imagined return (CMA-ES):	learned -0.04131
- Qualitative joint-goal tracking:
  > The arm reaches each random goal smoothly without overshoot. 
  > The learned weights act as extra damping (qd diagonal ≈ −0.2)
  > and a small push toward the goal (q − goal diagonal ≈ −0.1), 
  > with small biases that offset gravity. The gain is limited 
  > because most of each episode is spent moving at the smoothing speed limit, 
  > which no policy can beat.

## WorldModelDriver
- Predict: normalize [state, action] with norm.json, run dynamics.onnx on the CPU backend, then add the de-normalized delta to the state.
  
- Reanchor: snap the ghost to the real state every ghostInterval steps or when the joint RMSE exceeds 0.2 rad; softCorrection (0.05) also blends 5% of the real state in every step.
  
- Forecast: roll the model 100 steps ahead from the ghost state, run MuJoCo FK on each predicted pose to get the end-effector path, then restore qpos.
  
- Observation: the ghost moves faster than the real arm because it receives the raw policy action, while the real arm's command is smoothed. Those large action jumps never appear in the training data, so this is out of distribution for the model.

## IK + live target
- IK method: damped least squares, Δq = Jᵀ(JJᵀ + λ²I)⁻¹e with λ = 0.05, position only (3×7 Jacobian by finite differences via mj_kinematics). It runs 10 iterations per physics step, warm-starts from the previous solution, limits each step to 0.1 rad, and clamps to joint limits. The solution goes to ArmJointController.SetGoal() and the learned controller tracks it.
  
- Workspace / failure cases: inside the reach (~0.8 m) the end effector settles within a few millimetres of the sphere. When the sphere is moved quickly, the arm lags because commands are rate-limited. When the sphere is out of reach, DLS stretches the arm toward it and stays stable instead of oscillating.

## Design question
Why can rollout collection stay in joint space while deployment tracks a Cartesian target?

The world model and the controller only reason about joint states and joint commands: they learn how the arm responds to joint targets, not where those targets sit in space. The IK layer is the only part that knows about Cartesian space, and it converts the target position into a joint goal, which is exactly the kind of input the controller was trained on. As long as the IK goals fall inside the joint region covered by the random rollouts (home ± 35% of each joint range), the learned model and policy transfer directly to Cartesian tracking.