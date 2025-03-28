strategies = {'Baseline', 'Basic Aug', 'Advanced Aug', 'Selective Aug'};
accuracy = [66.754, 56.038, 52.646, 63.368];

figure;
bar(accuracy, 0.6);
set(gca, 'XTickLabel', strategies, 'FontSize', 12);
ylabel('Accuracy under Strong Perturbation (%)');
title('Model Robustness under Strong Perturbation');
ylim([0 80]);
grid on;

saveas(gcf, 'strong_perturbation_accuracy.png');