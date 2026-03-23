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

fprintf('\n--- Estrazione Matrici Strutturali (Rotazione JSZ) ---\n');

% 1. Ricreo la dinamica nello spazio latente X (Forma Canonica)
lam = lam_opt; r_inf = rinf_opt; N = 3; n_max = max(maturities);
K1_X   = diag(lam);
K0_X   = zeros(N, 1);
rho1_X = ones(N, 1);
rho0_X = r_inf;

% 2. Ricalcoliamo le equazioni di Riccati per trovare la matrice di rotazione
B_tilde = zeros(n_max, N);
B_tilde(1, :) = -rho1_X';
for n = 2:n_max
    B_tilde(n, :) = B_tilde(n-1, :) * (eye(N) + K1_X) - rho1_X';
end

B_X_obs = zeros(length(maturities), N);
for i = 1:length(maturities)
    B_X_obs(i, :) = -B_tilde(maturities(i), :) / maturities(i);
end

% Matrice di rotazione JSZ (M1) e Sigma ruotata
M1 = W * B_X_obs;
Sigma_X = M1 \ Sigma_dec;

A_tilde = zeros(n_max, 1);
A_tilde(1) = -rho0_X;
for n = 2:n_max
    conv_term = 0.5 * (B_tilde(n-1, :) * Sigma_X) * (B_tilde(n-1, :) * Sigma_X)';
    A_tilde(n) = A_tilde(n-1) + B_tilde(n-1, :) * K0_X + conv_term - rho0_X;
end

A_X_obs = zeros(length(maturities), 1);
for i = 1:length(maturities)
    A_X_obs(i) = -A_tilde(maturities(i)) / maturities(i);
end

% Vettore di traslazione JSZ (M0)
M0 = W * A_X_obs;

% 3. ROTAZIONE NELLO SPAZIO OSSERVABILE (Fattori PCA)
% Matrici K0 e K1 sotto misura Q
K1_Q = M1 * K1_X / M1;
%K0_Q = M0 - K1_Q * M0; % Poiché K0_X = 0
K0_Q = -K1_Q * M0; %fix

% 4. ESTRAZIONE DELL'EQUAZIONE DEL TASSO OMBRA (Short Rate s_t)
% Il tasso a 1 mese (n=1) definisce s_t. Prendiamo A e B per n=1 nello spazio X.
A1_X = -A_tilde(1);    % Corrisponde a r_inf
B1_X = -B_tilde(1, :); % Corrisponde a rho1_X'

% Ruota l'equazione del tasso a 1 mese nello spazio PCA
rho1 = (B1_X / M1)';
rho0 = A1_X - B1_X * (M1 \ M0);

fprintf('Parametri estratti con successo\n');

fprintf('\nEquazione shadow rate:\n')
fprintf('rho0: %.4f (in decimale mensile)\n', rho0)
fprintf('rho1: %.4f  %.4f  %.4f\n', rho1(1), rho1(2), rho1(3))

% Verifica: il tasso short medio dovrebbe essere vicino alla media del 3M
shadow_preLB = rho0 + factors_dec * rho1;
fprintf('Shadow rate medio pre-LB: %.2f%%\n', mean(shadow_preLB) * 1200)
fprintf('Media yield 3M pre-LB:    %.2f%%\n', mean(yields_preLB(:,1)))

% =========================================================================
%%  5. RICOSTRUZIONE CURVA E MISURA DEGLI ERRORI
% =========================================================================

% Ricostruisco la curva per calcolare l'errore di misura (sigma_e)
[~, Y_hat_dec, A_pca, B_pca] = jsz_obj_function(theta_opt, yields_dec, ...
                                        factors_dec, maturities, W, Sigma_dec);

% Porto le stime in percentuale annualizzata
Y_hat_pct = Y_hat_dec * 1200; 
errori_preLB = yields_preLB - Y_hat_pct;

% RMSE per ogni scadenza
rmse_bps = sqrt(mean(errori_preLB.^2)) * 100;
fprintf('\nRMSE In-Sample pre-LB (in basis points):\n');
disp(array2table(rmse_bps, 'VariableNames', {'y3M','y6M','y1Y','y2Y','y3Y','y5Y','y7Y','y10Y'}));

% Volatilità dell'errore di misurazione globale (sigma_e)
sigma_e_bps = std(errori_preLB(:)) * 100;
fprintf('Volatilita erore di misura (sigma_e): %.2f bps\n', sigma_e_bps);

% Plot del Fit
figure;
plot(maturities, mean(yields_preLB), 'bo-', 'LineWidth', 1.5); hold on;
plot(maturities, mean(Y_hat_pct), 'r*--', 'LineWidth', 1.5);
title('Fit Medio della Curva dei Rendimenti (Pre-LB)');
xlabel('Scadenze (mesi)'); ylabel('Rendimento (%)');
legend('Curva Osservata', 'Modello JSZ', 'Location', 'best');
grid on;


% =========================================================================
%%  6. GRID SEARCH PER IL LOWER BOUND E ESTRAZIONE STATI LATENTI
% =========================================================================

fprintf('\n--- Inizio Grid Search per r_LB (Scala Decimale Mensile) ---\n');

% 1. Convertiamo in Decimale Mensile TUTTI gli input del filtro
yields_LB_dec = yields_LB / 1200; %Yields_LB

r_LB_grid_bps = -120:10:0; %il Deposit Facility rate storicamente è sceso sotto lo zero in area euro, escludo quindi che il lower bound sia positivo 
r_LB_grid_dec = (r_LB_grid_bps / 10000) / 12;

sigma_e_dec = sigma_e_bps / 10000 / 12; 
R_mat_dec = eye(length(maturities)) * (sigma_e_dec^2);

logL_results = zeros(length(r_LB_grid_bps), 1);

% Parametri P in scala decimale (fix)
Phi_P = A_AR;   
mu_P_dec  = KP0 / 1200; 

% Stati iniziali rigorosamente in decimale
idx_preLB = sum(dates_monthly <= date_split);
P_init_dec = factors_all(idx_preLB, :)' / 1200; 
%V_init = eye(3) * 1e-6; non so quale sia meglio come tolleranza da
%testare, approfondire
V_init = eye(3) * 1e-3;

% Lanciamo il ciclo usando Sigma_dec (Blocco 4)
for i = 1:length(r_LB_grid_bps)
    current_rLB_dec = r_LB_grid_dec(i);
    fprintf('Testando r_LB = %3d bps... ', r_LB_grid_bps(i));
    
    [logL, ~] = latent_run_EKF_shadow(yields_LB_dec, P_init_dec, V_init, mu_P_dec, ...
                Phi_P, Sigma_dec, K0_Q, K1_Q, rho0, rho1, current_rLB_dec, maturities, R_mat_dec);
                        
    logL_results(i) = logL;
    fprintf('Log-Likelihood: %.4f\n', logL);
end

[~, idx_opt] = max(logL_results);
r_LB_opt_bps = r_LB_grid_bps(idx_opt);
r_LB_opt_dec = r_LB_grid_dec(idx_opt); 

fprintf('\n=== RISULTATO OTTIMIZZAZIONE LOWER BOUND ===\n');
fprintf('Il Lower Bound (r_LB) ottimale stimato è: %d bps\n', r_LB_opt_bps);

fprintf('\nRicalcolo l''EKF con il Lower Bound ottimale (%d bps)...\n', r_LB_opt_bps);
[~, P_latenti_ottimali_dec] = latent_run_EKF_shadow(yields_LB_dec, P_init_dec, V_init, mu_P_dec, ...
                Phi_P, Sigma_dec, K0_Q, K1_Q, rho0, rho1, r_LB_opt_dec, maturities, R_mat_dec);

figure('Name', 'Log-Likelihood r_LB');
plot(r_LB_grid_bps, logL_results, 'bo-', 'MarkerFaceColor', 'b');
xline(r_LB_opt_bps, 'r--', 'LineWidth', 1.5);
title('Log-Likelihood al variare del Lower Bound');
xlabel('r_{LB} (basis points)'); ylabel('Log-likelihood');
grid on;


%=== RISULTATO OTTIMIZZAZIONE LOWER BOUND ===
% Il Lower Bound (r_LB) ottimale stimato è: -70 bps
% Ricalcolo l'EKF con il Lower Bound ottimale (-70 bps)...



% =========================================================================
% FORZATURA DEL LOWER BOUND (Test ad esempio 0lb di Wu-Xia ZLB) vedere se
% lo shadow rate esplode in negativo mettendo dei LB irrealistici
% =========================================================================
%r_LB_opt_bps = -65; % Imponiamo un pavimento a 0 bps (oppure metti 25)
%r_LB_opt_dec = (r_LB_opt_bps / 10000) / 12; % Riconvertiamo in decimale
%r_LB_opt_pct = r_LB_opt_bps / 100;          % Riconvertiamo in percentuale

%fprintf('\nATTENZIONE: Forzo il Lower Bound a %d bps per testare lo Shadow Rate!\n', r_LB_opt_bps);

% Ora lanciamo l'EKF finale che estrarrà i fattori basandosi su questo LB finto!
%[~, P_latenti_ottimali_dec] = latent_run_EKF_shadow(yields_LB_dec, P_init_dec, V_init, mu_P_dec, ...
%                Phi_P, Sigma_dec, K0_Q, K1_Q, rho0, rho1, r_LB_opt_dec, maturities, R_mat_dec);

% =========================================================================
%%  7. ESTRAZIONE E PLOT DEL VERO SHADOW RATE (Fattori EKF)
% =========================================================================

fprintf('\n--- Estrazione e Plot del VERO Shadow Rate (Fattori Latenti) ---\n');

% Splicing fattori: unione dei fattori in decimale, pre LB e post LB
% estratti dal filtro

factors_preLB_dec = factors_all(1:idx_preLB, :) / 1200; 
P_history_true_dec = [factors_preLB_dec; P_latenti_ottimali_dec];

% Calcoliamo lo shadow rate in decimale
% (s_t = rho0 + rho1 * P_t)
shadow_rate_vero_dec = rho0 + P_history_true_dec * rho1;

% Riconverto alla fine in percentuale per il grafico
shadow_rate_vero_pct = shadow_rate_vero_dec * 1200;
r_LB_opt_pct = r_LB_opt_bps / 100;

% Il tasso "fisico" (modello) è semplicemente il massimo tra lo shadow rate e il pavimento.
% r_t=max(r_LB,s_t)
short_rate_model_pct = max(shadow_rate_vero_pct, r_LB_opt_pct);

figure('Name', 'VERO Shadow Rate (Latente)', 'Position', [100, 100, 800, 500]);

% A. Disegniamo le curve del modello
plot(dates_monthly, yields_all(:,1), 'k-', 'LineWidth', 1.5); hold on;
plot(dates_monthly, shadow_rate_vero_pct, 'b-', 'LineWidth', 1.5);
plot(dates_monthly, short_rate_model_pct, 'r--', 'LineWidth', 1.5);

% B. Linee di riferimento (Pavimento, Zero, e Split Date)
yline(r_LB_opt_pct, 'g-', 'LineWidth', 2, 'Label', sprintf('Lower Bound (%.0f bps)', r_LB_opt_bps));
yline(0, 'k:', 'LineWidth', 1); 
xline(date_split, 'm--', 'LineWidth', 1.5, 'Label', 'Inizio Filtro EKF');

title('VERO Shadow Rate (Fattori EKF) vs Tasso Osservato');
legend('Tasso Osservato (3M AAA)', 'Shadow Rate Libero (s_t)', ...
       'Tasso Troncato Modello (max(s_t, r_{LB}))', 'Location', 'best');
ylabel('Rendimento (%)');
grid on;

% Sovrapponiamo altre variabili macro cruciali per contestualizzare lo Shadow Rate.

% 1. Il tasso a 10 anni (yields_all(:,8)): mostra come, nonostante i tassi a breve 
%    fossero bloccati sul pavimento, la BCE usasse il QE per abbassare i tassi a lungo termine.
plot(dates_monthly, yields_all(:,8), 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'DisplayName', 'Tasso 10Y AAA');

% 2. ECB Deposit Facility Rate (DFR): Il VERO pavimento politico deciso dalla Banca Centrale.
%    Confrontarlo con il r_LB_opt (stimato dal modello) ci fa capire quanto "premio di scarsità" 
%    c'era sui titoli AAA rispetto ai tassi ufficiali.
ecb_df = readmatrix("ecb_df.xlsx","Range", "B2:B260");
plot(dates_monthly, squeeze(ecb_df(:)), 'c-', 'LineWidth', 1.5, 'DisplayName', 'ECB DFR');

% 3. Tasso Interbancario a 3 mesi (Euribor/OIS): Il mercato reale senza premio AAA.
irt3m = readmatrix("irt3m.xlsx","Range", "C144:C402");
plot(dates_monthly, squeeze(irt3m(:)), 'y-', 'LineWidth', 1.5, 'DisplayName', 'Interbank 3M');

% Nota finale per il grafico:
legend('show', 'Location', 'southwest');