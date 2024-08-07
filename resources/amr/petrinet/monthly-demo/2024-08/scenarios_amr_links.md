Scenarios:
•	6Mo Eval S2: Q1b (ingest the model with whatever process makes the most sense, and do unit tests), Q2a
•	12Mo Eval S1: Q1a.ii, Q2c
•	12Mo Eval S2 : Q1b, Q2
Scenario Materials:
1.	(6Mo Eval Scenarios) https://github.com/DARPA-ASKEM/program-milestones/tree/main/6-month-milestone/evaluation
2.	(12Mo Eval Scenarios): https://github.com/DARPA-ASKEM/program-milestones/tree/main/12-month-milestone/evaluation
For any simulation questions, you may simplify model configuration and just use reasonable values. We want to do some simulation with these models to demonstrate that the stratified or transformed versions give reasonable results that make sense, and can be used to compare outcomes for different age groups, or compare results before and after stratification. Ignore any calibration questions or comparison with other questions outside of this subset.

AMRs

6 Month Eval S2

Q1b.(i): SIDARTHE, original parameters https://github.com/siftech/funman/blob/d909110cf4c82b62569d37979703c36260909c41/resources/amr/petrinet/mira/models/BIOMD0000000955_askenet.json 

Q1b.(ii): SIDARTHE, updated (step function) parameters
•	https://github.com/siftech/funman/blob/d909110cf4c82b62569d37979703c36260909c41/resources/amr/petrinet/mira/models/scenario2_a.json#L1029
•	https://github.com/siftech/funman/blob/d909110cf4c82b62569d37979703c36260909c41/resources/amr/petrinet/mira/models/scenario2_a_beta_scale_var.json#L1029
•	https://github.com/siftech/funman/blob/d909110cf4c82b62569d37979703c36260909c41/resources/amr/petrinet/mira/models/scenario2_a_beta_scale_var_fixed.json#L1029

Q2a. SIDARTHE-V: https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.01/scenario2_sidarthe_v.json


12 Month Eval S1: 
Q1a.ii:
(0): Original SEIRHD (may not be necessary): https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario1_base.json
(1): SEIRHD: modify beta  beta*(1-epsilon_m*c_m): https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario1_1_ii_1.json
(2): SEIRHD: time-varying beta: https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario1_1_ii_2.json
(3): SEIRHD: stratify by masking: https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario1_1_ii_3.json
Q2c: SIRHD (removed E component for simplicity), stratified into 18 age groups: https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario1_2_sirhd_age.json 
Reference for Q2c (may not need to use this): SIRHD not yet stratified by age: https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario1_2_sirhd.json


12 Month Eval S2 : 
Base model: https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario2_base.json
Q1b: SEIRHD with multiple vaccine components: https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario2_1_b.json 
Q2: SEIRHD extension to include contact matrices – age-stratified 0-9, 10-19, 20-29: https://github.com/gyorilab/mira/blob/main/notebooks/evaluation_2023.07/eval_scenario2_2_a.json


