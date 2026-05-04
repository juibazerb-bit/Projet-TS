%% =====================================================================
%  Projet TS1 - Objectif n°2
%  Effet Doppler avec parasites musicaux : filtrage du signal
%  "son_2.wav" puis estimation de la vitesse du vehicule.
%
%  Strategie de filtrage :
%       - filtre passe-bande IIR (Butterworth) centre autour de f0
%       - largeur choisie pour englober la plage Doppler attendue
%
%  Auteurs : Nom Prenom
% =====================================================================

clear; close all; clc;

%% --- 1. Parametres -------------------------------------------------------
f0 = 4000;          % frequence emise (Hz)
c  = 340;           % celerite du son (m/s)
d  = 6;             % distance perpendiculaire (m)
t1 = 5;             % instant de passage (s)

%% --- 2. Lecture du signal ----------------------------------------------
[x, Fs] = audioread('son_2.wav');
x = x(:,1);
N  = length(x);
t  = (0:N-1).' / Fs;

%% --- 3. Spectrogramme avant filtrage -----------------------------------
Nwin    = round(0.05 * Fs);
Noverl  = round(0.9  * Nwin);
Nfft    = 2^nextpow2(4*Nwin);

figure('Name','Avant filtrage');
spectrogram(x, hamming(Nwin), Noverl, Nfft, Fs, 'yaxis');
title('son\_2.wav - signal brut (musique + Doppler)');
colormap turbo;

%% --- 4. Filtre passe-bande de Butterworth ------------------------------
%   Plage utile : on encadre largement la frequence Doppler attendue.
%   v ~ 30 m/s -> Df ~ +/- 350 Hz autour de 4 kHz, on prend une marge.
fLow  = 3500;       % Hz
fHigh = 4500;       % Hz
ordre = 6;

[bz, az] = butter(ordre, [fLow fHigh]/(Fs/2), 'bandpass');

% Affichage de la reponse du filtre
figure('Name','Reponse du filtre');
freqz(bz, az, 4096, Fs);
title(sprintf('Filtre passe-bande Butterworth ordre %d : [%d ; %d] Hz', ...
              ordre, fLow, fHigh));

% Filtrage a phase nulle (pour ne pas decaler le signal)
xf = filtfilt(bz, az, x);

%% --- 5. Spectrogramme apres filtrage -----------------------------------
figure('Name','Apres filtrage');
spectrogram(xf, hamming(Nwin), Noverl, Nfft, Fs, 'yaxis');
ylim([3 5]);
title('son\_2.wav apres filtrage passe-bande [3,5 - 4,5] kHz');
colormap turbo;

%% --- 6. Suivi de la frequence instantanee ------------------------------
Nseg = round(0.05 * Fs);
hop  = round(Nseg/4);
nbSeg = floor((N - Nseg)/hop) + 1;

t_f  = zeros(nbSeg, 1);
f_t  = zeros(nbSeg, 1);

w    = hamming(Nseg);
NfftLoc = 2^nextpow2(8*Nseg);
freqAxe = (0:NfftLoc-1).' * Fs/NfftLoc;
mask    = (freqAxe >= fLow) & (freqAxe <= fHigh);

for k = 1:nbSeg
    idx     = (k-1)*hop + (1:Nseg);
    seg     = xf(idx) .* w;
    X       = abs(fft(seg, NfftLoc));
    Xband   = X .* mask;
    [~, im] = max(Xband);
    if im > 1 && im < NfftLoc
        a = X(im-1); b = X(im); c2 = X(im+1);
        delta = 0.5*(a - c2)/(a - 2*b + c2);
    else
        delta = 0;
    end
    f_t(k) = (im - 1 + delta) * Fs / NfftLoc;
    t_f(k) = ((k-1)*hop + Nseg/2) / Fs;
end

f_t_lisse = movmean(f_t, 9);

figure('Name','Frequence instantanee apres filtrage');
plot(t_f, f_t, '.', 'Color',[0.7 0.9 0.7]); hold on;
plot(t_f, f_t_lisse, 'g-', 'LineWidth', 1.5);
yline(f0,'k--','f_0 = 4 kHz');
xlabel('t (s)'); ylabel('f (Hz)');
title('Frequence instantanee f(t) apres filtrage');
grid on;

%% --- 7. Vitesse radiale v_r(t) -----------------------------------------
v_r = c * (f0 ./ f_t_lisse - 1);

figure('Name','Vitesse radiale v_r(t)');
plot(t_f, v_r, 'r-','LineWidth',1.5); hold on;
yline(0,'k--'); xline(t1,'k:','t_1 = 5 s');
xlabel('t (s)'); ylabel('v_r (m/s)');
title('Vitesse radiale de la voiture - son\_2 filtre');
grid on;

%% --- 8. Estimation de la vitesse ---------------------------------------
masque_loin = (t_f < t1 - 2) | (t_f > t1 + 2);
v_estime_asymp = mean(abs(v_r(masque_loin)));

modele = @(p,tt) p(1)^2 .* (tt - t1) ./ sqrt(d^2 + p(1)^2*(tt - t1).^2);
opts   = optimset('Display','off');
p_fit  = lsqcurvefit(modele, v_estime_asymp, t_f, v_r, 0, 100, opts);
v_estime_fit = p_fit(1);

f_max = max(f_t_lisse); f_min = min(f_t_lisse);
v_estime_extr = c * (f_max - f_min) / (f_max + f_min);

fprintf('\n--- Estimation apres filtrage (son_2.wav) ---\n');
fprintf('Asymptotique  : %.2f m/s   (%.1f km/h)\n', v_estime_asymp, v_estime_asymp*3.6);
fprintf('Ajustement    : %.2f m/s   (%.1f km/h)\n', v_estime_fit  , v_estime_fit  *3.6);
fprintf('f_max / f_min : %.2f m/s   (%.1f km/h)\n', v_estime_extr , v_estime_extr *3.6);

% Comparaison modele
v_r_modele = modele(p_fit, t_f);
figure('Name','Comparaison modele/mesure');
plot(t_f, v_r, 'r-','LineWidth',1.2); hold on;
plot(t_f, v_r_modele,'k--','LineWidth',1.5);
xlabel('t (s)'); ylabel('v_r (m/s)');
title(sprintf('Modele : v = %.1f m/s', v_estime_fit));
legend('Mesure filtree','Modele'); grid on;
