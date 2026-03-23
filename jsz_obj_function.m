% =========================================================================
%% FUNZIONI LOCALI
% =========================================================================
function [SSE, Y_hat, A_pca, B_pca] = jsz_obj_function(theta, Y_actual, P_actual, maturities, W, Sigma)
    
    % Input:
    % theta: I 4 parametri che stiamo ottimizzando [lam1; lam2; lam3; r_inf]
    % Y_actual: I rendimenti veri del mercato (Tx8)
    % P_actual: I fattori veri estratti con la PCA (Tx3)
    % maturities: Le scadenze in mesi [3, 6, 12, ...]
    % W: La matrice dei pesi della PCA (collega i rendimenti ai fattori)
    % Sigma: La matrice di Cholesky (volatilità storica dei fattori)

    lam = theta(1:3);
    r_inf = theta(4);

    % Penalità: gli autovalori devono essere strettamente negativi per la stazionarietà
    % e devono essere distinti (altrimenti la matrice di rotazione diventa singolare)
    
    if any(lam >= 0) || length(unique(round(lam, 4))) < 3
        SSE = 1e10; Y_hat = []; A_pca = []; B_pca = []; return;
        % Se l'algoritmo fminsearch prova a esplorare parametri fisicamente impossibili, 
        % gli restituiamo un errore enorme (SSE = 1e10) per farlo "scappare" da quell'area.
        % Regola 1: I lambda (lam) devono essere strettamente negativi per la stazionarietà.
        % Regola 2: I lambda devono essere tutti diversi tra loro, altrimenti le matrici 
        %           diventano singolari e non invertibili.
    end

    N = 3; % Numero di fattori
    n_max = max(maturities); % Orizzonte massimo (es. 120 mesi)

    % Dinamica Q per i fattori latenti (Forma Canonica JSZ)
    K1_X = diag(lam);
    K0_X = zeros(N, 1);
    rho1_X = ones(N, 1);
    rho0_X = r_inf;

    % 1) Equazioni di Riccati per calcolare i loading B sui fattori latenti
    B_tilde = zeros(n_max, N);
    B_tilde(1, :) = -rho1_X'; % Condizione iniziale per il tasso a 1 mese
    for n = 2:n_max
        % Ricorsione: l'esposizione al mese 'n' dipende da quella al mese 'n-1'
        B_tilde(n, :) = B_tilde(n-1, :) * (eye(N) + K1_X) - rho1_X';
    end

    % Estraiamo i B latenti solo per le scadenze che osserviamo
    % Dividiamo per 'n' per convertire il prezzo del bond in rendimento (yield).
    B_X_obs = zeros(length(maturities), N);
    for i = 1:length(maturities)
        n = maturities(i);
        B_X_obs(i, :) = -B_tilde(n, :) / n;
    end

    % 2) Matrice di rotazione JSZ (Da spazio latente X a spazio PCA)
    M1 = W * B_X_obs; % Dimensioni: 3x8 * 8x3 = 3x3

    if cond(M1) > 1e10 %scarto parametri impsobbili matrici mal condizionate
         SSE = 1e10; Y_hat = []; A_pca = []; B_pca = []; return;
    end

    % Sigma ruotata nello spazio latente
    Sigma_X = M1 \ Sigma;

    % 3) Equazioni di Riccati per A (include il termine di convessità)
    A_tilde = zeros(n_max, 1);
    A_tilde(1) = -rho0_X;
    for n = 2:n_max
        % Termine convessità: 0.5 * B_{n-1} * Sigma_X * Sigma_X' * B_{n-1}'
        conv_term = 0.5 * (B_tilde(n-1, :) * Sigma_X) * (B_tilde(n-1, :) * Sigma_X)';
        A_tilde(n) = A_tilde(n-1) + B_tilde(n-1, :) * K0_X + conv_term - rho0_X;
    end

    A_X_obs = zeros(length(maturities), 1);
    for i = 1:length(maturities)
        n = maturities(i);
        A_X_obs(i) = -A_tilde(n) / n;
    end

    % 4) Traslazione M0 per i fattori PCA
    M0 = W * A_X_obs; 

    % 5) Mappatura analitica finale: convertiamo A_X e B_X nello spazio osservabile PCA
    A_pca = A_X_obs - B_X_obs * (M1 \ M0);
    B_pca = B_X_obs / M1;

    % Generiamo i rendimenti impliciti del modello
    Y_hat = zeros(size(Y_actual));
    for t = 1:size(Y_actual, 1)
        Y_hat(t, :) = (A_pca + B_pca * P_actual(t, :)')';
    end

    % Funzione obiettivo: Somma degli Errori Quadrati (SSE)
    errors = Y_actual - Y_hat;
    SSE = sum(sum(errors.^2));
end

    % =====================================================================
    % MATEMATICA DEL MODELLO (JSZ 2011 Framework)
    % =====================================================================
    % 1. TASSO A BREVE (Short Rate):
    %    r_t = rho0_X + rho1_X' * X_t
    %
    % 2. DINAMICA DEI FATTORI LATENTI SOTTO MISURA Q (Risk-Neutral):
    %    X_t = K0_X + (I + K1_X) * X_{t-1} + Sigma_X * eps_t^Q
    %    (JSZ impone K0_X = 0, rho1_X = vettore di 1, K1_X = diag(lambda))
    %
    % 3. EQUAZIONI DI RICCATI PER I PREZZI (A_tilde, B_tilde):
    %    Il prezzo del bond è P_{n,t} = exp(A_tilde_n + B_tilde_n * X_t)
    %    - B_tilde_n = B_tilde_{n-1} * (I + K1_X) - rho1_X'
    %    - A_tilde_n = A_tilde_{n-1} + B_tilde_{n-1}*K0_X + 0.5*(B_tilde_{n-1}*Sigma_X)*(...)' - rho0_X
    %
    % 4. CONVERSIONE IN RENDIMENTI (Yields):
    %    y_{n,t} = A_X_n + B_X_n * X_t
    %    Dove: A_X_n = -A_tilde_n / n
    %          B_X_n = -B_tilde_n / n
    %
    % 5. ROTAZIONE NELLO SPAZIO OSSERVABILE (PCA):
    %    I fattori PCA sono: P_t = W * Y_t
    %    Poiché Y_t = A_X + B_X * X_t, allora:
    %    P_t = W * A_X + W * B_X * X_t  =>  P_t = M0 + M1 * X_t
    %
    % 6. LOADINGS FINALI (Mappatura X_t -> P_t):
    %    B_pca = B_X * M1^{-1}
    %    A_pca = A_X - B_pca * M0
    %    Modello finale: Y_hat = A_pca + B_pca * P_t
    % =====================================================================