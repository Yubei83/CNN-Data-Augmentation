models = {'Baseline', 'Basic Aug', 'Advanced Aug', 'Selective Aug'};
accuracy = [99.576, 99.174, 99.33, 99.60]; 

figure;
bar(accuracy);
set(gca, 'XTickLabel', models, 'FontSize', 12);
ylabel('Validation Accuracy (%)');
title('Validation Accuracy Comparison across Augmentation Strategies');
ylim([98 100]);  
grid on;

saveas(gcf, 'validation_accuracy_comparison.png');