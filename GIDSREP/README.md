# Artifact of Paper: R+R: From Claims to Crashes: A Systematic Re-evaluation of Graph-Based Network Intrusion Detection Systems

In this repo, we provide experiment results of this paper, evaluation scripts and code to facilitate the reproducibility and the replicability of our work.

We provide an introduction of each experiment in the Artifact Appendix.



## Folders

We implement our study as a framework, GIDSREP。Given raw data as input, GIDSREP generates experimental results through four modules: (1) the Data Processing Module, which cleans, formats, and partitions data for each model; (2) the Detection Assessment Module, which evaluates detection performance using default and tuned hyperparameters; (3) the Robustness Assessment Module, which tests model resilience against adversarial attacks; and (4) the Efficiency Assessment Module, which measures space and time performance.

The Efficiency Assessment Module is not provided in this code base. It mainly uses memory_profiler to measure the space occupancy

Our experiment evaluated six models,include **ARGUS**, **EULER**, **VGRNN**, **PIKCHU** and **Anomal-E** .

### 1-DATE_PROCESSING

For the subsequent modules cleans, formats, and partitions data.

### 2-DETECTION_ASSESSMENT

It includes RQ1-3 in the paper, the **Reproduction** experiments on the LANL and OPTC datasets, and the **Replication** experiments on the cic_2017,LANL,OPTC and real-world datasets.

### 3-ROBUSTNESS_ASSESSMENT

It includes RQ5 in the paper. We tested the resilience of three models, **ARGUS**,**EULER** and **VGRNN**, against adversarial attacks on the **LANL**,**OPTC** and **real-world** (0501) datasets.



## Setup

The experimental environment refers to the Settings in the respective code repositories of ARGUS, EULER, VGRNN, PIKACHU and Anomal-E.



## Possible Variance:

Note that there will be some **variance** in the benchmark results.  For example, the float-point computation in different GPU may not always be the same, leading to a different round-up in some results.   However, you are still expected to observe similar results.

