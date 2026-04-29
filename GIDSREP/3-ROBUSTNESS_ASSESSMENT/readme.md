# ROBUSTNESS ASSESSMENT

### This module corresponds to RQ5 in the paper: How resilient are these models to adversarial attacks?

Through simulating white-box adversarial attacks, we apply perturbations aimed at evading detection to the testing set.

### Run the experiment

Each model setting selects the optimal parameters from the previous RQS. For any model and any dataset, there are similar operations.  For example, the experiments of the **ARGUS** model on the **OPTC** dataset are as follows:

1. Enter the relevant directory

   ```shell
   cd ARGUS
   unzip model_argus.zip
   cp model_save_OPTC_ARGUS.pkl ./optc/
   cd optc
   ```

2. Fill in the file location of the output of the **OPTC** dataset in the data processing module in line 19 of the **load_optc.py** file under the loaders folder

3. Start the adversarial attack

   ```python
   python eva_attack.py
   ```

PS: Our dataset used in this module is n_9000.