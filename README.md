# GTSRB Data Augmentation and Robustness Analysis Project

This project investigates the effect of data augmentation strategies on the robustness of traffic sign classification using the GTSRB dataset and AlexNet.

The baseline model achieves high validation accuracy (>99%), so the focus is shifted to robustness under synthetic perturbations. Four augmentation strategies are implemented:

- Baseline (no augmentation)  
- Basic augmentation (flip, translation, scaling)  
- Advanced augmentation (adds rotation, brightness, noise)  
- Selective augmentation (mild rotation and translation, based on class-level difficulty)

Each strategy is trained over 5 independent runs. Robustness is evaluated under two conditions:

- **Moderate**: blur (σ=1.0), small-angle rotation, gamma (0.9)  
- **Strong**: blur (σ=3.5), large rotation, gamma (0.4), salt-and-pepper noise (3%)

Evaluation includes validation accuracy, robustness accuracy, standard deviation, and variance.

## File Structure

- `baseline_run*.m`, `baseline_run*_robustness_*.m`: baseline training and testing  
- `aug_basic_run*.m`, `aug_basic_robustness_run*.m`: basic augmentation  
- `aug_adv_run*.m`, `aug_adv_robustness_run*_strong.m`: advanced augmentation  
- `selective_aug_run*.m`, `selective_aug_robustness_*_run*.m`: selective augmentation  
- `generate_robustness_test.m`, `generate_robustness_mid_test.m`: test set generation  
- `accuracy_table.m`: computes mean, std, variance  
- `validation_accuracy_comparison.m`, `strong_perturbation_accuracy.m`, `robustness_baseline_selective_boxplot.m`: plotting

Implemented in MATLAB. Dataset: GTSRB.
