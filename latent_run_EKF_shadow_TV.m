% -------------------------------------------------------------------------
% FUNZIONE: Extended Kalman Filter con approssimazione Wu & Xia (2016)
% VERSIONE TIME-VARYING LOWER BOUND (TV-LB)
% Segue Lemke & Vladu (2017), equazione (13)
% -------------------------------------------------------------------------
% DIFFERENZA RISPETTO ALLA VERSIONE BASE:
% r_LB_vec è ora un vettore Tx1. Al tempo 't', il mercato osserva il 
% pavimento current_LB = r_LB_vec(t) e assume (per evitare che veda nel futuro)
% che rimarrà tale per tutte le scadenze future 'h'.
% -------------------------------------------------------------------------
function [logL, P_filtered_history] = latent_run_EKF_shadow_TV(Y_obs_dec, P0_dec, V0, ...
    mu_P_dec, Phi_P, Sigma_dec, K0_Q, K1_Q, rho0_dec, rho1, r_LB_vec, maturities, R_mat_dec)
    
    T     = size(Y_obs_dec, 1);
    N     = length(P0_dec);
    n_max = max(maturities);
    
    % matrice di transizione sotto Q
    A_Q   = eye(N) + K1_Q;        
    SigSig = Sigma_dec * Sigma_dec';
    
    % --- Precomputa sigma_h^Q: deviazione standard del shadow rate h passi avanti ---
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
        % ESTRAZIONE DEL LOWER BOUND CORRENTE
        % ==============================================================
        current_LB = r_LB_vec(t);
        
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
        
        % Tasso short corrente: r_0 = max(s_0, current_LB)
        s_0 = rho0_dec + rho1' * P_pred;
        r_0 = max(s_0, current_LB);
        
        for jj = 1:J_mats
            n = maturities(jj);
            % --- A. Calcolo dello yield del modello ---
            mu_P_curr = P_pred;
            sum_fwd   = 0;
            
            % calcolo l'aspettativa per ogni singolo mese 'h' nel futuro
            for h = 1:n-1
                % Valore atteso dei fattori tra 'h' mesi sotto misura Q
                mu_P_curr = K0_Q + A_Q * mu_P_curr;   % E^Q[P_{t+h}]
                % Valore atteso dello shadow rate tra 'h' mesi
                mu_s_h    = rho0_dec + rho1' * mu_P_curr;
                
                % Standardizziamo usando il current_LB
                arg_h     = (mu_s_h - current_LB) / sigma_h(h);
                
                % Approssimazione analitica dell'aspettativa troncata
                H_val     = arg_h * normcdf(arg_h) + normpdf(arg_h);
                sum_fwd   = sum_fwd + sigma_h(h) * H_val;
            end
            % Il rendimento totale: usiamo current_LB anche per la parte a pronti
            Y_hat_dec(jj) = (r_0 + (n-1) * current_LB + sum_fwd) / n;
            
            % --- Jacobiano analitico: H=dY/dP_pred ---
            % Contributo di r_0: d(max(s_0,LB))/dP_k = rho1(k) se s_0 > current_LB
            dr0_dP = rho1 * (s_0 > current_LB);   % vettore Nx1
            
            % Contributo della somma
            mu_P_curr = P_pred;
            A_pow     = eye(N);                   % A_Q^0
            dsum_dP   = zeros(N, 1);
            
            for h = 1:n-1
                A_pow     = A_Q * A_pow;           % A_Q^h
                mu_P_curr = K0_Q + A_Q * mu_P_curr;
                mu_s_h    = rho0_dec + rho1' * mu_P_curr;
                
                % Di nuovo, la derivata dipende dal current_LB
                arg_h     = (mu_s_h - current_LB) / sigma_h(h);
                
                % Chain rule con la CDF
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
            logL = -1e10; return; % esci se matrice esplode
        end
        
        % Aggiorno la Log-Likelihood totale
        logL = logL - 0.5 * (J_mats * log(2*pi) + log(detF) + err' * (F \ err));
        
        % Kalman Gain e Update Stato
        K_gain = V_pred * H' / F;
        P_pred = P_pred + K_gain * err;
        P_filtered_history(t, :) = P_pred';
        
        % Formula di Joseph per la covarianza
        IKH    = eye(N) - K_gain * H;
        V_pred = IKH * V_pred * IKH' + K_gain * R_mat_dec * K_gain';
        V_pred = 0.5 * (V_pred + V_pred');
    end
end