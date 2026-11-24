using UnityEngine;


[DefaultExecutionOrder(-10)]
public class ArmJointController : MonoBehaviour
{
    public ArmAgent agent;
    public float holdSeconds = 3f;

    public Vector2[] jointRanges;

    public float[] homeTargets;

    [Range(0f, 1f)]
    public float randomScale = 0.35f;

    public bool emitActions = true;

    float[] target;
    float hold;
    bool ready;

    public float[] CurrentTarget => target;
    
    // For fix BAD QACC
    public float maxCommandStep = 0.05f;
    float[] command;  
    
    // IK
    public bool randomGoals = true;
    
    public void SetGoal(float[] goal)
    {
        for (int i = 0; i < target.Length; i++)
            target[i] = Mathf.Clamp(goal[i], jointRanges[i].x, jointRanges[i].y);
    }

    void Start()
    {
        if (agent == null) agent = GetComponent<ArmAgent>();

        int n = agent.ActionSize;
        if (jointRanges == null || jointRanges.Length != n
                                || homeTargets == null || homeTargets.Length != n)
        {
            Debug.LogError($"ArmJointController: jointRanges and homeTargets must both have {n} entries.");
            enabled = false;
            return;
        }

        target = new float[n];
        PlaceTarget(true);
        
        // Fix Bad QACC
        command = new float[n];
        
        hold = holdSeconds;
        ready = true;
    }

    void FixedUpdate()
    {
        if (!ready || !agent.Ready) return;
        
        if (randomGoals)
        {
            hold -= Time.fixedDeltaTime;
            if (hold <= 0f)
            {
                PlaceTarget(false);
                hold = holdSeconds;
            }
        }

        // Fix Bad QACC
        for (int i = 0; i < command.Length; i++)
        {
            command[i] += Mathf.Clamp(target[i] - command[i], -maxCommandStep, maxCommandStep);
        }
        
        if (emitActions)
            agent.SetAction(command);
    }

    public void ActionFrom(float[] destination)
    {
        System.Array.Copy(target, destination, target.Length);
    }

    public void PlaceTarget(bool home)
    {
        for (int i = 0; i < target.Length; i++)
        {
            float lo = jointRanges[i].x;
            float hi = jointRanges[i].y;

            if (home)
                target[i] = homeTargets[i];
            else
            {
                float span = (hi - lo) * 0.5f * randomScale;
                target[i] = homeTargets[i] + Random.Range(-span, span);
            }
            target[i] = Mathf.Clamp(target[i], lo, hi);
        }
    }
}
