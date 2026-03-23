% =========================================================================
%  FAVAR – Shadow Rate EA - Replication/Extension of 
%  Wolfgang Lemke, Andreea Liliana Vladu 
%  Working Paper Series Below the zero lower bound: a shadow-rate term structure model for the euro area
%  No 1991 / January 2017
%  
%  Author: Lorenzo Pisa
% =========================================================================

clear all; close all;
addpath(genpath(pwd))

% =========================================================================
%%  1. DATA LOADING
% =========================================================================

data    = readmatrix("data/aaa_yelds_curve.xlsx",       "Range", "C16:J5520");
dates_str = readcell("data/aaa_yelds_curve.xlsx", "Range", "A16:A5520");
dates_all = datetime(dates_str, 'InputFormat', 'yyyy-MM-dd');


T = timetable(dates_all, data(:,1), data(:,2), data(:,3), data(:,4), ...
              data(:,5), data(:,6), data(:,7), data(:,8), ...
    'VariableNames', {'y10Y','y1Y','y2Y','y3M','y3Y','y5Y','y6M','y7Y'});

% Resample a fine mese (ultima osservazione del mese)
T_monthly = retime(T, 'monthly', 'lastvalue');

% Ordino le colonne per scadenza crescente
yields_all = [T_monthly.y3M, T_monthly.y6M, T_monthly.y1Y, T_monthly.y2Y, ...
          T_monthly.y3Y, T_monthly.y5Y, T_monthly.y7Y, T_monthly.y10Y];
dates_monthly = T_monthly.dates_all;

%data view 
figure;
plot(T_monthly.dates_all, yields_all);
legend('3M','6M','1Y','2Y','3Y','5Y','7Y','10Y');
title('AAA Yield Curve - Euro Area');
ylabel('Yield (%)');
grid on;

% Split pre-LB / LB
date_split   = datetime(2011, 10, 1);
%date_split   = datetime(2014,02, 1);
idx_preLB    = dates_monthly <= date_split;
idx_LB       = dates_monthly >  date_split;

yields_LB    = yields_all(idx_LB, :);
yields_preLB = yields_all(idx_preLB, :);
dates_preLB  = dates_monthly(idx_preLB);
dates_LB     = dates_monthly(idx_LB);

fprintf('Osservazioni pre-LB: %d\n', sum(idx_preLB))
fprintf('Osservazioni LB:     %d\n', sum(idx_LB))
fprintf('Totale:              %d\n', length(dates_monthly))

% Plot dati completi con linea di split
figure;
plot(dates_monthly, yields_all, 'LineWidth', 1);
xline(date_split, 'r--', 'LineWidth', 2, 'Label', 'Lower Bound Start');
legend('3M','6M','1Y','2Y','3Y','5Y','7Y','10Y', ...
       'Location', 'northeast');
title('AAA Yield Curve - Euro Area (2004-2026)');
ylabel('Yield (%)'); grid on;

%il plot qui funziona, 1. OK

% =========================================================================
%%  2. PCA - solo pre-LB
% =========================================================================
% Uso solo il pre-LB pe evitare distorsioni date dal lower bound.
% La media del pre-LB viene usata per demeaning anche fuori campione.
% Troviamo i pesi W usando PCA (che fa il demeaning internamente per trovare le componenti)

[coeff, score, ~, ~, explained] = pca(yields_preLB);

fprintf('Varianza spiegata PC1: %.2f%%\n', explained(1))
fprintf('Varianza spiegata PC2: %.2f%%\n', explained(2))
fprintf('Varianza spiegata PC3: %.2f%%\n', explained(3))
fprintf('Totale primi 3:        %.2f%%\n', sum(explained(1:3)))

% se sopra al 95% ok

% Matrice di proiezione (3x8)
W = coeff(:, 1:3)';             

% Nel modello affine, i fattori osservabili devono 
% contenere il livello assoluto dei tassi, quindi non sottraggo la media
factors_all   = yields_all * W';      % Proiezione pura sui tassi grezzi (Tx3)

% Splitto i fattori corretti
factors_preLB = factors_all(idx_preLB, :);
factors_LB    = factors_all(idx_LB, :);

% =========================================================================
%%  3. VAR sotto misura P - solo pre-LB
% =========================================================================
% Stimo il VAR(1) (OLS) sui fattori pre-LB.
% Modello del apper: Delta(P_t) = KP0 + KP1*P_{t-1} + Sigma*eps
% cioè P_t = KP0 + (I+KP1)*P_{t-1} + Sigma*eps
% Quindi AR=(I + KP1), e non KP1 da sola!.

N     = 3;
T_pre = size(factors_preLB, 1);

% Variabile dipendente P_t, regressori [1, P_{t-1}]
Y_var = factors_preLB(2:end, :);                      % (T-1) x 3
X_var = [ones(T_pre-1,1), factors_preLB(1:end-1, :)]; % (T-1) x 4

% OLS
Beta  = (X_var' * X_var) \ (X_var' * Y_var); % 4x3

% Estraggo i parametri
KP0   = Beta(1, :)';        % 3x1 drift
A_AR  = Beta(2:end, :)';    % 3x3 matrice AR = (I + KP1)
KP1   = A_AR - eye(N);      % 3x3 mean reversion (inverto l'equazioni sopra)

% Residui e Sigma tramite Cholesky
resid     = Y_var - X_var * Beta;
Sigma_cov = (resid' * resid) / (T_pre - 2);
Sigma     = chol(Sigma_cov, 'lower');  % triangolare inferiore, do un ordine di impatto agli shock 1:livello 2:pendenza 3:curvatura, i 3 fattori estratti

% =========================================================================
%%  3b. VERIFICA VAR 
% =========================================================================

% --- Verifica 1: autovalori (gia' fatto, ma lo ripetiamo in modo chiaro) ---
ev = eig(A_AR);
fprintf('Autovalori matrice AR:\n')
disp(ev)
fprintf('Moduli (devono essere tutti < 1):\n')
disp(abs(ev))

% --- Verifica 2: fit in-sample del VAR sui fattori ---
% Il VAR prevede P_t dato P_{t-1}
% Confrontiamo la previsione con i fattori osservati
T_pre     = size(factors_preLB, 1);
X_var     = [ones(T_pre-1,1), factors_preLB(1:end-1,:)];
Y_hat_var = X_var * Beta;   % previsioni un passo avanti

figure;
titles = {'Fattore 1 - Livello', 'Fattore 2 - Pendenza', 'Fattore 3 - Curvatura'};
for i = 1:3
    subplot(3,1,i)
    plot(dates_preLB(2:end), factors_preLB(2:end,i), 'b-', 'LineWidth', 1.5)
    hold on
    plot(dates_preLB(2:end), Y_hat_var(:,i), 'r--', 'LineWidth', 1)
    legend('Osservato','Previsto VAR')
    title(titles{i}); grid on;
end

% --- Verifica 3: residui del VAR ---
resid = factors_preLB(2:end,:) - Y_hat_var;

% R-squared per ogni equazione
SS_res = sum(resid.^2, 1);
SS_tot = sum((factors_preLB(2:end,:) - mean(factors_preLB(2:end,:))).^2, 1);
R2     = 1 - SS_res ./ SS_tot;
fprintf('\nR-squared per equazione VAR:\n')
fprintf('  Fattore 1 (livello):   %.4f\n', R2(1))
fprintf('  Fattore 2 (pendenza):  %.4f\n', R2(2))
fprintf('  Fattore 3 (curvatura): %.4f\n', R2(3))

% Autocorrelazione dei residui - test di Ljung-Box a lag 1
fprintf('\nAutocorrelazione residui a lag 1:\n')
for i = 1:3
    r1 = corr(resid(2:end,i), resid(1:end-1,i));
    fprintf('  Fattore %d: %.4f\n', i, r1)
end

% Plot residui
figure;
for i = 1:3
    subplot(3,1,i)
    plot(dates_preLB(2:end), resid(:,i), 'k-')
    yline(0, 'r--')
    title(['Residui VAR - ' titles{i}]); grid on;
end


%NOTA sui valori di autocorrelazione:
% 
% Autocorrelazione residui a lag 1:
%  Fattore 1: 0.3820
%  Fattore 2: 0.1534
%  Fattore 3: 0.0442

% Mantengo il VAR(1) nonostante lo 0.38: 
% - overfitting su un campione pre-LB solo 80 osservazioni circa 
% - Per rimanere fedele al paper di Lemke & Vladu 2017 
% - semplicità del modello nelle seizoni sucessive

% NOTE on residual autocorrelation:
% 
% Residual autocorrelation at lag 1:
%  Factor 1: 0.3820
%  Factor 2: 0.1534
%  Factor 3: 0.0442
% Keeping the VAR(1) specification despite the 0.38 value due to: 
% - Risk of overfitting on a short pre-LB sample (only ~80 observations) 
% - Consistency with the Lemke & Vladu (2017) paper 
% - Model simplicity and tractability in the subsequent sections

% =========================================================================
%%  4. STIMA PARAMETRI Q - JSZ esatto (4 parametri)
% =========================================================================
maturities = [3, 6, 12, 24, 36, 60, 84, 120];  % scadenza in mesi

% Conversione in tassi mensili decimali per evitare 
% l'esplosione del termine di convessità nelle equazioni di Riccati.
yields_dec  = yields_preLB / 1200;
factors_dec = factors_preLB / 1200;
Sigma_dec   = Sigma / 1200; 

% Parametri iniziali [lambda1; lambda2; lambda3; r_inf]
% sono la velocità di mean reversion sotto Q
% I lambda devono essere negativi (mean-reverting) e 
% distinti per non creare matrici singoalri non invertibili.

theta0 = [-0.01; -0.05; -0.10; 0.04/12];
%theta0 = [-0.02; -0.07; -0.12; 0.03/12];

% valori a priori (educated guesses):
% -0.01: il Livello ha una persistenza altissima (shock quasi permanenti)
% -0.05: la Pendenza è un po' meno persistente
% -0.10: la Curvatura ha una mean-reversion molto rapida (gli shock rientrano in fretta)
% r_inf è il tasso a lungo termine (es. 4% = 0.04 annualizzato -> diviso 12)



% Opzioni di ottimizzazione (fminsearch)
options = optimset('Display','iter','MaxFunEvals', 10000, ...
                   'MaxIter', 10000, 'TolFun', 1e-8, 'TolX', 1e-8);

% la funzione obiettivo
obj_fun = @(theta) jsz_obj_function(theta, yields_dec, factors_dec, ...
                                    maturities, W, Sigma_dec);

% Lancio Ottimizzazione
fprintf('\n--- Inizio Ottimizzazione JSZ (Cross-Section) ---\n');
[theta_opt, fval] = fminsearch(obj_fun, theta0, options);

% Estrazione parametri
lam_opt  = theta_opt(1:3);
rinf_opt = theta_opt(4);

fprintf('\n=== PARAMETRI Q OTTIMALI (JSZ) ===\n');
fprintf('Lambda 1: %.4f\n', lam_opt(1));
fprintf('Lambda 2: %.4f\n', lam_opt(2));
fprintf('Lambda 3: %.4f\n', lam_opt(3));
fprintf('r_inf (annualizzato): %.2f%%\n', rinf_opt * 1200);

% Validità e Plausibilità (External Check):
% I lambda stimati per l'Euro Area AAA sono circa (-0.0076, -0.0474, -0.0718).
% Se confrontati con la Tabella 2 di JSZ (2011), che riportava per i dati USA 
% valori pari a (-0.0024, -0.0481, -0.0713), si nota che sono quasi identici. 
% CONCLUSIONE: L'Area Euro ha una struttura di pricing estremamente simile a quella USA 
% in termini di avversione al rischio e velocità di rientro degli shock (sotto Q).

% =========================================================================
%%  4b. ESTRAZIONE PARAMETRI Q E MAPPATURE
% =========================================================================

% =========================================================================
%%  5. RICOSTRUZIONE CURVA E MISURA DEGLI ERRORI
% =========================================================================

% =========================================================================
%%  6. GRID SEARCH PER IL LOWER BOUND E ESTRAZIONE STATI LATENTI
% =========================================================================

% =========================================================================
%%  7. ESTRAZIONE E PLOT DEL VERO SHADOW RATE (Fattori EKF)
% =========================================================================
