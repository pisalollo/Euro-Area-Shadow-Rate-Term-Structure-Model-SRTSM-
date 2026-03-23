% -------------------------------------------------------------------------
% FUNZIONE: Extended Kalman Filter con approssimazione Wu & Xia (2016)
% Segue Lemke & Vladu (2017), equazione (13)
% -------------------------------------------------------------------------
% MATEMATICA DEL FILTRO EKF (Extended Kalman Filter)
%
% 1. PREDICT STEP (Dinamica sotto misura P - Mondo Reale):
%    P_{t|t-1} = mu^P + Phi^P * P_{t-1|t-1}
%    V_{t|t-1} = Phi^P * V_{t-1|t-1} * (Phi^P)' + Sigma*Sigma'
%
% 2. MEASUREMENT STEP (Approssimazione analitica Wu & Xia):
%    Il tasso forward atteso è troncato al Lower Bound (LB). 
%    Si usa la formula chiusa per l'aspettativa di una Normale troncata:
%    y_n = (1/n) * [ r_0 + (n-1)*LB + sum_{h=1}^{n-1} sigma_h * g( (mu_{s,h} - LB)/sigma_h ) ]
%    Dove g(x) = x * Phi(x) + phi(x)  (Phi=CDF, phi=PDF della Normale)
%
% 3. JACOBIANO ANALITICO (Matrice H):
%    Invece di derivate numeriche lente e instabili, usiamo la Chain Rule:
%    H = d(y_n) / d(P) = (1/n) * [ d(r_0)/d(P) + sum d(g)/d(P) ]
%    La derivata di g(x) rispetto a mu_s è semplicemente la CDF: Phi(x)!
%
% 4. UPDATE STEP (Formula di Joseph):
%    P_{t|t} = P_{t|t-1} + K * (Y_{obs} - Y_{hat})
%    V_{t|t} = (I - K*H) * V_{t|t-1} * (I - K*H)' + K * R * K'
% -------------------------------------------------------------------------

function [logL, P_filtered_history] = latent_run_EKF_shadow(Y_obs_dec, P0_dec, V0, ...
    mu_P_dec, Phi_P, Sigma_dec, K0_Q, K1_Q, rho0_dec, rho1, r_LB_dec, maturities, R_mat_dec)

    T     = size(Y_obs_dec, 1);
    N     = length(P0_dec);
    n_max = max(maturities);
    
    % matrice di transizione sotto Q
    A_Q   = eye(N) + K1_Q;        
    SigSig = Sigma_dec * Sigma_dec';

    % --- Precomputa sigma_h^Q: deviazione standard del shadow rate h passi avanti ---
    % Questa dipende solo dai parametri, non dallo stato -> calcolata una volta sola
    % Var(s_{t+h}) = rho1' * Var(P_{t+h}) * rho1
    % Var(P_{t+h}) cresce ricorsivamente: Var_h = A_Q * Var_{h-1} * A_Q' + SigSig
    Var_P_h  = zeros(N, N);
    sigma_h  = zeros(n_max, 1);
    for h = 1:n_max
        Var_P_h  = A_Q * Var_P_h * A_Q' + SigSig;
        % Deviazione standard dello shadow rate al tempo t+h
        sigma_h(h) = sqrt(max(rho1' * Var_P_h * rho1, 1e-14));
    end
 
    P_filtered_history = zeros(T, N);
    P_pred = P0_dec;
    V_pred = V0;
    Q_mat  = SigSig;
    logL   = 0;

    for t = 1:T

        % ==============================================================
        % PREDICT: propaga stato con VAR sotto P
        % ==============================================================
        P_pred = mu_P_dec + Phi_P * P_pred;
        V_pred = Phi_P * V_pred * Phi_P' + Q_mat;
        % Forza simmetria e definita positività
        V_pred = 0.5 * (V_pred + V_pred');
        [V_ev, V_ed] = eig(V_pred);
        V_pred = V_ev * diag(max(real(diag(V_ed)), 1e-12)) * V_ev';

        % ==============================================================
        % WU & XIA: calcola yields modello e Jacobiano analiticamente
        % ==============================================================
        J_mats   = length(maturities);
        Y_hat_dec = zeros(J_mats, 1);
        H         = zeros(J_mats, N);

        % Tasso short corrente: r_0 = max(s_0, LB)
        s_0 = rho0_dec + rho1' * P_pred;
        r_0 = max(s_0, r_LB_dec);

        for jj = 1:J_mats
            n = maturities(jj);

            % --- A. Calcolo dello yield del modello (Equazione 13 L&V) ---
            % y_n = (1/n) * [r_0 + (n-1)*LB + sum_{h=1}^{n-1} sigma_h * H(arg_h)]
            mu_P_curr = P_pred;
            sum_fwd   = 0;
            
            % calcolo l'aspettativa per ogni singolo mese 'h' nel futuro
            for h = 1:n-1
                % Valore atteso dei fattori tra 'h' mesi sotto misura Q
                mu_P_curr = K0_Q + A_Q * mu_P_curr;   % E^Q[P_{t+h}]
                % Valore atteso dello shadow rate tra 'h' mesi
                mu_s_h    = rho0_dec + rho1' * mu_P_curr;
                % Standardizziamo la variabile (distanza dal Lower Bound)
                arg_h     = (mu_s_h - r_LB_dec) / sigma_h(h);
                % Approssimazione analitica dell'aspettativa troncata
                H_val     = arg_h * normcdf(arg_h) + normpdf(arg_h);
                sum_fwd   = sum_fwd + sigma_h(h) * H_val;
            end
            % Il rendimento totale è la media del tasso r_0 e delle aspettative future
            Y_hat_dec(jj) = (r_0 + (n-1) * r_LB_dec + sum_fwd) / n;

            % --- Jacobiano analitico: H=dY/dP_pred ---
            % Contributo di r_0: d(max(s_0,LB))/dP_k = rho1(k) se s_0 > LB
            % altrimenti 0

            dr0_dP = rho1 * (s_0 > r_LB_dec);   % vettore Nx1

            % Contributo della somma: usa dH/darg = Phi(arg) e chain rule
            mu_P_curr = P_pred;
            A_pow     = eye(N);                   % A_Q^0
            dsum_dP   = zeros(N, 1);

            for h = 1:n-1
                A_pow     = A_Q * A_pow;           % A_Q^h
                mu_P_curr = K0_Q + A_Q * mu_P_curr;
                mu_s_h    = rho0_dec + rho1' * mu_P_curr;
                arg_h     = (mu_s_h - r_LB_dec) / sigma_h(h);

                % La CDF rappresenta la "probabilità neutrale al rischio" di essere sopra il LB.
                % d(sigma_h * H_val)/d(mu_s_h) = sigma_h * Phi(arg_h) / sigma_h = Phi(arg_h)
                % d(mu_s_h)/d(P_pred) = rho1' * A_Q^h  =>  (A_Q^h)' * rho1
                dsum_dP = dsum_dP + normcdf(arg_h) * (A_pow' * rho1);
            end
            
            H(jj, :) = (dr0_dP + dsum_dP)' / n;
        end

        % ==============================================================
        % UPDATE: correggi con l'osservazione
        % ==============================================================
        err = Y_obs_dec(t, :)' - Y_hat_dec;

        F = H * V_pred * H' + R_mat_dec;
        F = 0.5 * (F + F') + 1e-10 * eye(J_mats);  % simmetria + regolarizzazione

        detF = det(F);
        if detF <= 0 || ~isfinite(detF)
            logL = -1e10; return; %esci se matrice esplode
        end
        
        %aggiorno la Log_likeihood totale
        logL = logL - 0.5 * (J_mats * log(2*pi) + log(detF) + err' * (F \ err));
        
        % Kalman Gain (Quanto peso diamo all'errore vs al modello)
        K_gain = V_pred * H' / F;

        % Aggiornamento finale dello Stato Latente (P_t|t)
        P_pred = P_pred + K_gain * err;
        P_filtered_history(t, :) = P_pred';

        % Formula di Joseph: numericamente più stabile di (I-KH)*V
        IKH    = eye(N) - K_gain * H;
        V_pred = IKH * V_pred * IKH' + K_gain * R_mat_dec * K_gain';
        V_pred = 0.5 * (V_pred + V_pred');
    end
end