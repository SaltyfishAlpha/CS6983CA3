using System.Linq;
using Mujoco;
using UnityEngine;

public class ArmAgent : MonoBehaviour
{
    public int JointCount;
    public int StateSize;
    public int ActionSize;

    public MjActuator gripper;
    public MjBody endEffectorBody;
    public Transform endEffector;
    public MjHingeJoint[] joints;
    public MjActuator[] actuators;

    float[] action;
    float[] state;
    bool ready;

    public float[] CurrentAction => action;
    public bool Ready => ready;

    void Awake()
    {
        if (JointCount <= 0) JointCount = 7;
        StateSize = JointCount * 2;
        ActionSize = JointCount;
        
        InitializeJointsAndActuators();

        action = new float[ActionSize];
        state = new float[StateSize];

        ready = true;
        
        Debug.Log($"ArmAgent: {joints.Length} joints, {actuators.Length} actuators");
    }

    void FixedUpdate()
    {
        if (!ready) return;
        ApplyAction();
    }

    public void InitializeJointsAndActuators()
    {
        joints = new MjHingeJoint[JointCount];
        actuators = new MjActuator[JointCount];

        for (int i = 0; i < JointCount; i++)
        {
            joints[i] = FindComponent<MjHingeJoint>("joint" + (i + 1));
            actuators[i] = FindComponent<MjActuator>("actuator" + (i + 1));
        }

        endEffectorBody = FindComponent<MjBody>("link7");
        endEffector = endEffectorBody != null ? endEffectorBody.transform : null;
    }

    public void SetAction(float[] targets)
    {
        int n = Mathf.Min(targets.Length, ActionSize);
        for (int i = 0; i < n; i++)
        {
            action[i] = targets[i];
        }
    }

    public void ApplyAction()
    {
        for (int i = 0; i < ActionSize; i++)
        {
            actuators[i].Control = action[i];
        }
    }

    public float[] Observe()
    {
        for (int i = 0; i < JointCount; i++)
        {
            state[i] = (float)joints[i].RawConfiguration;
            state[JointCount + i] = joints[i].Velocity;
        }
        
        return state;
    }

    public void CopyState(float[] destination)
    {
        Observe();
        System.Array.Copy(state, destination, StateSize);
    }

    private T FindComponent<T>(string name) where T : Component
    {
        var found = FindObjectsByType<T>(FindObjectsSortMode.None).FirstOrDefault(c =>
            c.gameObject.name == name
        );
        if (found == null)
            Debug.LogError($"ArmAgent: could not find {typeof(T).Name} named '{name}'");
        return found;
    }
}
