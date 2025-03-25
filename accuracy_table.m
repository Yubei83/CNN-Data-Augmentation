Model = {
    'Baseline'; 'Baseline'; 'Baseline'; 'Baseline'; 'Baseline'; 'Baseline Avg';
    'Basic Aug'; 'Basic Aug'; 'Basic Aug'; 'Basic Aug'; 'Basic Aug'; 'Basic Aug Avg';
    'Adv Aug'; 'Adv Aug'; 'Adv Aug'; 'Adv Aug'; 'Adv Aug'; 'Adv Aug Avg';
    'Selective Aug'; 'Selective Aug'; 'Selective Aug'; 'Selective Aug'; 'Selective Aug'; 'Selective Avg'
};

Run = {
    1; 2; 3; 4; 5; 'Avg';
    1; 2; 3; 4; 5; 'Avg';
    1; 2; 3; 4; 5; 'Avg';
    1; 2; 3; 4; 5; 'Avg'
};

ValAcc = [
    99.47; 99.27; 99.73; 99.68; 99.73; 99.576;
    98.67; 99.11; 99.23; 99.27; 99.59; 99.174;
    99.46; 99.44; 99.69; 99.30; 98.76; 99.33;
    99.47; 99.64; 99.71; 99.74; 99.44; 99.6
];

MidAcc = [
    99.42; 99.37; 99.74; 99.53; 99.81; 99.574;
    NaN; NaN; 98.56; NaN; NaN; NaN;
    NaN; 99.3; NaN; NaN; NaN; NaN;
    99.63; 99.6; 99.74; 99.79; 99.12; 99.576
];

StrongAcc = [
    49.09; 67.33; 66.33; 71.88; 79.14; 66.754;
    NaN; NaN; 49.51; NaN; NaN; NaN;
    NaN; 51.42; NaN; NaN; NaN; NaN;
    73.3; 66.14; 66.19; 48.21; 63; 63.368
];

T = table(Model, Run, ValAcc, MidAcc, StrongAcc, ...
    'VariableNames', {'Model_Type', 'Run', 'Validation_Accuracy', 'Mid_Perturbation_Accuracy', 'Strong_Perturbation_Accuracy'});

disp(T);
