# Evotask

Evotask is an in-development tensorflow library designed to perform parallelizable, reproducible agent-environment simulations where the behaviors of both the agent and the environment may be parameterized and, potentially, evolved. Above and beyond standard fitness assessment and evolutionary optimization, which are dramatically accelerated by the library, this library is ideal for experiments involving multi-task learning, co-evolution, and dynamic task development.

Evotask is used in "Reinforcement learning beyond the Bellman equation: Exploring critic objectives using evolution", by Abe Leite, Madhavun Candadai, and Eduardo J. Izquierdo. This paper is available in the proceedings of the Artificial Life 2020 conference [here](https://doi.org/10.1162/isal_a_00338). In this context, evotask is used to simulate a parallelized version of the inverted pendulum task and perform the DDPG and actor-fitting algorithms for a population of agents.

Evotask depends on tensorflow (for computation) and OpenAI gym (for rendering). Other code in this repository depends on scipy for statistics and pyplot, pandas, and seaborn to produce figures.

Please be aware that all of the evotask code included in this repository is at an alpha stage of development. The tools included with the library developed dramatically during our experiments for ALife 2020, and I anticipate that they will continue to do so for some time. If you are interested in making use of or contributing to the evotask project beyond simply replicating our experiments for this paper, please contact me (Abe Leite) at [abrahamjleite@gmail.com](mailto:abrahamjleite@gmail.com) and I will provide the up-to-date code to you. I would be delighted to hear from and/or work with you!
