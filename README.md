# Reinforcement learning beyond the Bellman equation: Exploring critic objectives using evolution
### Abe Leite, Madhavun Candadai, and Eduardo J. Izquierdo

This repository includes code to enable replication and extension of our 2020 Artificial Life paper, available in the proceedings of the conference [here](https://doi.org/10.1162/isal_a_00338).

We have included `evotask`, an internally-developed framework for evolving agents to perform a number of flexible and potentially evolvable tasks. Evotask is in rapid development according to our needs and research questions and is not intended as a stable library; if you find it useful, however, please reach out and I can keep you up-to-date with any changes or improvements we make. If you've spent some time with the code and have thoughts about it, I would also appreciate hearing about any comments or criticisms.

We have also included a number of scripts that call evotask functions to replicate our 2020 ALife experiments, as well as scripts to analyze the results of those experiments. Our own results will also be available under the Release tab of this repository on GitHub, which is better suited to hosting binary files.

## Dependencies:

This work has been primarily tested on linux since it is very compute-intensive. Please alert us if there are any difficulties running it on other platforms.

Evotask depends on tensorflow (for computation) and OpenAI gym (for rendering). Other code in this repository depends on scipy for statistics and pyplot, pandas, and seaborn to produce figures. To ensure that these are installed in your chosen python's site-packages, you may run

    python3 -m pip install tensorflow gym scipy matplotlib pandas seaborn

If you find any difficulties, you can also precisely replicate our environment by running (preferably in a venv so as to not interfere with your primary python installation)

    python -m pip install -r requirements.txt

If you are in a virtual environment, you may need to replace the invocations of `python3` in our shell scripts with `python`.

## Running:

To replicate our experiments, please run `replicate_experiments.sh`, or distribute the included commands as you see fit. This takes a significant amount of compute power -- around a week of compute time on a single Nvidia Tesla V100. After running this script, you will have a number of files in `alife-results/` including the sequences of rewards when developing critics through reinforcement learning and population search, the Q-map and training signal spectra of the critics, and the sequence of rewards when the critics are tasked with training 10 fresh actors from scratch each. These files are all included in `alife-results.tar.xz` and can be extracted from it.

To replicate our analysis, please run `replicate_analysis.sh`. This should take less than a minute on an average laptop computer. It will report the statistical tests we included in our paper and generate the figures (or the data components of the figures) for the paper.

## Contact:

These experiments have led me to refocus my remaining time studying at Indiana University on hybrid approaches to evolution and learning. If these ideas are of any interest to you, or if you have any questions about this work and its interpretation, please do not hesitate to contact me at [abrahamjleite@gmail.com](mailto:abrahamjleite@gmail.com).
