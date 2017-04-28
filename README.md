# reinforcement_learning
Contains different reinforcement learning algorithms, including visualisation and analysis tools:

1. tabular_q_learning.py: Implementation of tabular q learning in OpenAI gym's frozen lake environment. While learning, produces animation of the value function and the preferred action according to the greedy policy.

2. deep_q_learning.py: Implementation of deep q learning in OpenAI gym's mountain car environment. While learning, produces animation of the value function, contours of areas with constant greedy choice of action, and the current trajectory in state space.

3. discounted_REINFORCE.py: Implementation of discounted REINFORCE in OpenAI gym's frozen lake environment. Based on John Schulmans code at http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/pg-startercode.py. Iterates over discount factor, for each discount, records multiple learning curves and save them to file.

4. discounted_REINFORCE_evaluation.py: Evaluates data generated by 3., outputs leanring speed over discount factor, or learning curves with standard deviation.

5. data: sample data generated by 3. to run 4. on.
