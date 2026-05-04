%% =====================================================================
%  Projet TS1 - Objectif n°1
%  Effet Doppler : estimation de la vitesse d'un vehicule a partir
%  de l'enregistrement audio "son_1.wav".
%
%  Modele physique :
%       f(t) = f0 * c / (c + v_r(t))
%       v_r(t) = v^2 * (t - t1) / sqrt(d^2 + v^2*(t - t1)^2)
%
%  Auteurs : Nom Prenom
% =====================================================================

clear; close all; clc;

%% --- 1. Parametres physiques --------------------------------------------
f0 = 4000;          % frequence emise (Hz)
c  = 340;           % celerite du son dans l'air (m/s)
d  = 6;             % distance perpendiculaire a la route (m)
t1 = 5;             % instant du passage perpendiculaire (s)

%% --- 2. Lecture du fichier audio ----------------------------------------
fichier = 'son_1.wav';
[x, Fs] = audioread(fichier);
x = x(:,1);                         % on garde un seul canal si stereo
N  = length(x);
t  = (0:N-1).' / Fs;                % vecteur temps (s)

fprintf('Fichier  : %s\n', fichier);
fprintf('Fs       : %d Hz\n', Fs);
fprintf('Duree    : %.2f s\n', N/Fs);

%% --- 3. Spectrogramme (vue d'ensemble) ----------------------------------
Nwin    = round(0.05 * Fs);         % fenetre 50 ms
Noverl  = round(0.9  * Nwin);       % 90 % de recouvrement
Nfft    = 2^nextpow2(4*Nwin);       % zero-padding pour resolution

figure('Name','Spectrogramme du son_1');
spectrogram(x, hamming(Nwin), Noverl, Nfft, Fs, 'yaxis');
ylim([3 5]);                        % zoom autour de 4 kHz
title('Spectrogramme du signal son\_1.wav');
colormap turbo;

%% --- 4. Extraction de la frequence instantanee f(t) ---------------------
%   On parcourt le signal par tranches, on calcule la FFT et on cherche
%   le pic spectral dans une bande [3.5 ; 4.5] kHz.

Nseg = round(0.05 * Fs);            % duree d'une tranche (50 ms)
hop  = round(Nseg/4);               % avance entre tranches (75 % overlap)
nbSeg = floor((N - Nseg)/hop) + 1;

t_f  = zeros(nbSeg, 1);
f_t  = zeros(nbSeg, 1);

w    = hamming(Nseg);
NfftLoc = 2^nextpow2(8*Nseg);       % zero-padding generaux pour interpoler
freqAxe = (0:NfftLoc-1).' * Fs/NfftLoc;
mask    = (freqAxe >= 3500) & (freqAxe <= 4500);

for k = 1:nbSeg
    idx     = (k-1)*hop + (1:Nseg);
    seg     = x(idx) .* w;
    X       = abs(fft(seg, NfftLoc));
    Xband   = X .* mask;
    [~, im] = max(Xband);
    
    % Affinage du pic par interpolation parabolique
    if im > 1 && im < NfftLoc
        a = X(im-1); b = X(im); c2 = X(im+1);
        delta = 0.5*(a - c2)/(a - 2*b + c2);
    else
        delta = 0;
    end
    f_t(k) = (im - 1 + delta) * Fs / NfftLoc;
    t_f(k) = ((k-1)*hop + Nseg/2) / Fs;
end

%% --- 5. Lissage de la courbe de frequence -------------------------------
f_t_lisse = movmean(f_t, 7);

figure('Name','Frequence instantanee f(t)');
plot(t_f, f_t, '.', 'Color',[0.7 0.7 0.9]); hold on;
plot(t_f, f_t_lisse, 'b-', 'LineWidth', 1.5);
yline(f0, 'k--', 'f_0 = 4 kHz');
xlabel('t (s)'); ylabel('f (Hz)');
title('Evolution de la frequence percue par l''observateur');
legend('Mesure brute','Lissee','f_0','Location','best');
grid on;

%% --- 6. Vitesse radiale v_r(t) = c*(f0/f(t) - 1) ------------------------
v_r = c * (f0 ./ f_t_lisse - 1);

figure('Name','Vitesse radiale v_r(t)');
plot(t_f, v_r, 'r-', 'LineWidth', 1.5); hold on;
yline(0,'k--'); xline(t1,'k:','t_1 = 5 s');
xlabel('t (s)'); ylabel('v_r (m/s)');
title('Vitesse radiale de la voiture vue par l''observateur');
grid on;

%% --- 7. Estimation de la vitesse v du vehicule --------------------------
% Methode 1 : par la valeur asymptotique
%   loin du passage (|t - t1| >> d/v), v_r tend vers +/- v.
masque_loin = (t_f < t1 - 2) | (t_f > t1 + 2);
v_estime_asymp = mean(abs(v_r(masque_loin)));

% Methode 2 : ajustement non lineaire du modele
modele = @(p,tt) p(1)^2 .* (tt - t1) ./ sqrt(d^2 + p(1)^2*(tt - t1).^2);
p0 = v_estime_asymp;
opts = optimset('Display','off');
[p_fit, ~] = lsqcurvefit(modele, p0, t_f, v_r, 0, 100, opts);
v_estime_fit = p_fit(1);

% Methode 3 : par les frequences extremes (approche grossiere)
f_max = max(f_t_lisse);
f_min = min(f_t_lisse);
v_estime_extr = c * (f_max - f_min) / (f_max + f_min);

fprintf('\n--- Estimation de la vitesse moyenne du vehicule ---\n');
fprintf('Asymptotique  : %.2f m/s   (%.1f km/h)\n', v_estime_asymp, v_estime_asymp*3.6);
fprintf('Ajustement    : %.2f m/s   (%.1f km/h)\n', v_estime_fit  , v_estime_fit  *3.6);
fprintf('f_max / f_min : %.2f m/s   (%.1f km/h)\n', v_estime_extr , v_estime_extr *3.6);

%% --- 8. Comparaison modele / mesure -------------------------------------
v_r_modele = modele(p_fit, t_f);
figure('Name','Modele vs mesure');
plot(t_f, v_r, 'r-', 'LineWidth',1.2); hold on;
plot(t_f, v_r_modele, 'k--', 'LineWidth',1.5);
xlabel('t (s)'); ylabel('v_r (m/s)');
title(sprintf('Ajustement du modele : v = %.1f m/s', v_estime_fit));
legend('Mesure','Modele','Location','best');
grid on;
