using Mujoco;
using UnityEngine;

[DefaultExecutionOrder(-8)]
public class IkGoalProvider : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public Transform target;

    public int iterations = 10;
    public float damping = 0.05f;
    public float maxStep = 0.1f;
    public float fdEps = 1e-3f;
    public float tolerance = 0.002f;

    float[] q;
    double[] savedQ;
    Vector3[] J;
    bool ready;

    void Start()
    {
        if (agent == null) agent = GetComponent<ArmAgent>();
        if (goals == null) goals = GetComponent<ArmJointController>();

        int n = agent.JointCount;
        q = (float[])goals.homeTargets.Clone();
        savedQ = new double[n];
        J = new Vector3[n];
        ready = target != null;
        if (!ready) Debug.LogError("IkGoalProvider needs a target Transform.");
    }
    
    unsafe Vector3 ForwardKinematics(float[] joints, MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
    {
        for (int i = 0; i < agent.JointCount; i++)
            d->qpos[agent.joints[i].QposAddress] = joints[i];
        MujocoLib.mj_kinematics(m, d);
        return MjEngineTool.UnityVector3(d->xpos + 3 * agent.endEffectorBody.MujocoId);
    }
    
    unsafe void FixedUpdate()
    {
        if (!ready || !agent.Ready || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;

        var scene = MjScene.Instance;
        var m = scene.Model;
        var d = scene.Data;
        int n = agent.JointCount;

        // 1. save the real arm's state
        for (int i = 0; i < n; i++)
            savedQ[i] = d->qpos[agent.joints[i].QposAddress];

        for (int it = 0; it < iterations; it++)
        {
            // 2. where is the end effector now, and how far from the target
            Vector3 p = ForwardKinematics(q, m, d);
            Vector3 e = target.position - p;
            if (e.magnitude < tolerance) break;

            // 3. Jacobian by finite differences: column j = how the EE moves when joint j turns a little
            for (int j = 0; j < n; j++)
            {
                float old = q[j];
                q[j] = old + fdEps;
                J[j] = (ForwardKinematics(q, m, d) - p) / fdEps;
                q[j] = old;
            }

            // 4. A = J J^T + λ^2 I   (3x3)
            var A = Matrix4x4.zero;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    float sum = 0f;
                    for (int j = 0; j < n; j++) sum += J[j][r] * J[j][c];
                    A[r, c] = sum + (r == c ? damping * damping : 0f);
                }
            A[3, 3] = 1f; // pad to 4x4

            // 5. y = inv(A) e,   Δq = J^T y
            Vector3 y = A.inverse.MultiplyVector(e);
            for (int j = 0; j < n; j++)
            {
                float dq = Mathf.Clamp(Vector3.Dot(J[j], y), -maxStep, maxStep);
                q[j] = Mathf.Clamp(q[j] + dq, goals.jointRanges[j].x, goals.jointRanges[j].y);
            }
        }

        // 6. restore the real arm
        for (int i = 0; i < n; i++)
            d->qpos[agent.joints[i].QposAddress] = savedQ[i];
        MujocoLib.mj_kinematics(m, d);

        // 7. hand the joint goal to the controller
        goals.SetGoal(q);
    }
}