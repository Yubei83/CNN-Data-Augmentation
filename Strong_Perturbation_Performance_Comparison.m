baseline_strong = [49.09, 67.33, 66.33, 71.88, 79.14];
selective_strong = [73.3, 66.14, 66.19, 48.21, 63];

mean_baseline = mean(baseline_strong);
var_baseline = var(baseline_strong);
std_baseline = std(baseline_strong);

mean_selective = mean(selective_strong);
var_selective = var(selective_strong);
std_selective = std(selective_strong);

fprintf('\n=== Strong Perturbation Performance Comparison ===\n');
fprintf('Baseline  -> Mean: %.2f%% | Std: %.2f | Var: %.2f\n', mean_baseline, std_baseline, var_baseline);
fprintf('Selective -> Mean: %.2f%% | Std: %.2f | Var: %.2f\n', mean_selective, std_selective, var_selective);
