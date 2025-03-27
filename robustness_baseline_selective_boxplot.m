baseline_strong = [49.09, 67.33, 66.33, 71.88, 79.14];
selective_strong = [73.3, 66.14, 66.19, 48.21, 63];

data = [baseline_strong, selective_strong];
group = [repmat({'Baseline'}, 1, 5), repmat({'Selective'}, 1, 5)];

figure;
boxplot(data, group, 'Colors', 'k', 'Symbol', 'r+');
ylabel('Accuracy under Strong Perturbation (%)');
title('Robustness Accuracy Distribution (5 Runs)');
grid on;

saveas(gcf, 'robustness_baseline_selective_boxplot.png');
