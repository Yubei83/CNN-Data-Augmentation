# GTSRB Data Augmentation and Robustness Analysis Project

This project investigates the impact of various data augmentation strategies on the robustness of traffic sign image classification models under perturbations. The model is based on a fine-tuned AlexNet and trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset. Since the original model already achieves near-saturated validation accuracy (above 99%), the project shifts its focus to robustness under synthetic distortions rather than further improving accuracy on clean data.

Four types of data augmentation strategies were designed and implemented: no augmentation (baseline), basic augmentation, advanced augmentation, and selective augmentation. Each strategy was executed with five independent training runs, and robustness was evaluated under both moderate and strong perturbation conditions. In addition, a set of scripts was developed to calculate statistics, generate visualizations, and analyze performance metrics such as standard deviation.

Main files included in the project are as follows:

- Files starting with `baseline` correspond to baseline experiments without data augmentation, including five training scripts and their corresponding evaluation scripts under moderate and strong perturbations.

- Files starting with `aug_basic` implement the basic augmentation strategy, including training and evaluation scripts, as well as a robustness test under moderate perturbations.

- Files starting with `aug_adv` contain training and evaluation scripts for the advanced augmentation strategy, along with scripts for testing under strong perturbations.

- Files starting with `selective_aug` are related to the selective augmentation strategy, covering five training runs and evaluations under both moderate and strong perturbation conditions.

- The files `generate_robustness_test.m` and `generate_robustness_mid_test.m` are used to generate perturbed test sets for strong and moderate conditions, respectively.

- The script `accuracy_table.m` compiles all accuracy results across validation, moderate, and strong perturbation evaluations, and calculates mean, standard deviation, and variance.

- The scripts `strong_perturbation_accuracy.m`, `validation_accuracy_comparison.m`, and `Strong_Perturbation_Performance_Comparison.m` are used to analyze and visualize performance metrics across strategies, supporting the figures presented in the report (Figure 4 and Figure 5).

It is recommended to run the project using MATLAB R2024b. GPU acceleration is highly recommended to improve training efficiency. This project uses the GTSRB (German Traffic Sign Recognition Benchmark) dataset, available at:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
