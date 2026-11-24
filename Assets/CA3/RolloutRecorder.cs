using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class RolloutRecorder : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController controller;
    public int steps = 70000;

    readonly List<float[]> states = new();
    readonly List<float[]> actions = new();
    readonly List<float[]> nextStates = new();

    int n;
    float[] prevS;
    float[] prevA;
    bool saved;

    void Start()
    {
        if (agent == null) agent = GetComponent<ArmAgent>();
        if (controller == null) controller = GetComponent<ArmJointController>();
        
        states.Clear();
        actions.Clear();
        nextStates.Clear();

        n = 0;
        prevS = null;
        prevA = null;
        saved = false;
    }

    void FixedUpdate()
    {
        if (saved || !agent.Ready) return;

        float[] s = new float[agent.StateSize];
        agent.CopyState(s);

        if (prevS != null)
        {
            states.Add(prevS);
            actions.Add(prevA);
            nextStates.Add(s);
            n++;
            
            if (n % 5000 == 0)
                Debug.Log($"RolloutRecorder: {n}/{steps}, tracking error {TrackingError():F3} rad");
        }

        float[] a = new float[agent.ActionSize];
        System.Array.Copy(agent.CurrentAction, a, agent.ActionSize);
        prevS = s;
        prevA = a;
        
        if (n >= steps) Save();
    }

    float TrackingError()
    {
        float[] s = agent.Observe();
        float[] a = agent.CurrentAction;
        float sum = 0f;
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float e = s[i] - a[i];
            sum += e * e;
        }
        return Mathf.Sqrt(sum / agent.ActionSize);
    }

    void Save()
    {
        saved = true;

        var ci = System.Globalization.CultureInfo.InvariantCulture;
        var sb = new StringBuilder();

        for (int k = 0; k < states.Count; k++)
        {
            AppendRow(sb, states[k], ci);
            sb.Append(',');
            AppendRow(sb, actions[k], ci);
            sb.Append(',');
            AppendRow(sb, nextStates[k], ci);
            sb.Append('\n');
        }

        string dir = Path.Combine(Application.dataPath, "CA3", "Data");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "rollout.csv");
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"RolloutRecorder: saved {states.Count} rows to {path}");
    }

    static void AppendRow(StringBuilder sb, float[] v, System.IFormatProvider ci)
    {
        for (int i = 0; i < v.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(v[i].ToString("G9", ci));
        }
    }
}
