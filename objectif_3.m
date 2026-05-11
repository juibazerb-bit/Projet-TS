%% =====================================================================
%  Projet TS1 - Objectif n°3
%  Estimation de la vitesse d'une voiture aux 24 Heures du Mans
%  a partir de l'enregistrement "Le_Mans_2023.wav".
%
%  Specificites :
%   - la frequence emise n'est pas connue a priori (bruit moteur),
%   - la voiture passe tres vite => le decalage Doppler est important
%     et l'enregistrement est court,
%   - on travaille sur une raie spectrale dominante.
%
%  Auteurs : Nom Prenom
% =====================================================================

clear; close all; clc;

%% --- 1. Parametres -------------------------------------------------------
c  = 340;           % celerite du son (m/s)
% La distance d et l'instant de passage t1 sont a estimer visuellement
% sur le spectrogramme.

%% --- 2. Lecture du signal ----------------------------------------------
[x, Fs] = audioread('Le_Mans_2023.wav');
x = x(:,1);
N  = length(x);
t  = (0:N-1).' / Fs;

fprintf('Duree de l''enregistrement : %.2f s\n', N/Fs);

%% --- 3. Spectrogramme : reperage de la raie dominante -------------------
Nwin    = round(0.03 * Fs);     % fenetre courte : meilleure resolution temps
Noverl  = round(0.9  * Nwin);
Nfft    = 2^nextpow2(8*Nwin);

figure('Name','Spectrogramme Le Mans');
spectrogram(x, hamming(Nwin), Noverl, Nfft, Fs, 'yaxis');
title('Spectrogramme Le\_Mans\_2023.wav');


%   --> repondre visuellement aux questions suivantes avant de continuer :
%       . quelle bande de frequence contient la raie principale ?
%       . a quel instant t1 la voiture passe-t-elle devant le micro
%         (transition du saut de frequence) ?
%
%   On parametre alors la suite avec les valeurs lues sur le graphique.

fLow_band  = 400;          % a ajuster apres lecture du spectrogramme (Hz)
fHigh_band = 1000;         % idem
t1_lu      = 1.45 ; %N/Fs/2;       % par defaut : milieu de l'enregistrement

%% --- 4. Filtrage selectif autour de la raie principale -----------------
[bz, az] = butter(6, [fLow_band fHigh_band]/(Fs/2), 'bandpass');
xf = filtfilt(bz, az, x);

%% --- 5. Suivi de la frequence instantanee ------------------------------
Nseg = round(0.02 * Fs);   % tranche tres courte : passage tres rapide
hop  = round(Nseg/4);
nbSeg = floor((N - Nseg)/hop) + 1;

t_f  = zeros(nbSeg, 1);
f_t  = zeros(nbSeg, 1);
amp  = zeros(nbSeg, 1);

w    = hamming(Nseg);
NfftLoc = 2^nextpow2(8*Nseg);
freqAxe = (0:NfftLoc-1).' * Fs/NfftLoc;
mask    = (freqAxe >= fLow_band) & (freqAxe <= fHigh_band);

for k = 1:nbSeg
    idx     = (k-1)*hop + (1:Nseg);
    seg     = xf(idx) .* w;
    X       = abs(fft(seg, NfftLoc));
    Xband   = X .* mask;
    [pkVal, im] = max(Xband);
    if im > 1 && im < NfftLoc
        a = X(im-1); b = X(im); c2 = X(im+1);
        delta = 0.5*(a - c2)/(a - 2*b + c2);
    else
        delta = 0;
    end
    f_t(k) = (im - 1 + delta) * Fs / NfftLoc;
    t_f(k) = ((k-1)*hop + Nseg/2) / Fs;
    amp(k) = pkVal;
end

% On ne garde que les segments ou la raie est suffisamment energetique
seuil = 0.2 * max(amp);
keep  = amp > seuil;
t_f_v = t_f(keep);
f_t_v = f_t(keep);
f_t_lisse = movmean(f_t_v, 5);

figure('Name','Frequence instantanee Le Mans');
plot(t_f_v, f_t_v, '.', 'Color',[0.8 0.6 0.6]); hold on;
plot(t_f_v, f_t_lisse, 'r-', 'LineWidth', 1.5);
xlabel('t (s)'); ylabel('f (Hz)');
title('Frequence instantanee de la raie dominante');
grid on;

%% --- 6. Estimation de la vitesse par les frequences extremes ------------
%   Loin du passage : f_appr = f0*c/(c-v),  f_eloign = f0*c/(c+v)
%   donc :
%       v = c * (f_max - f_min) / (f_max + f_min)
%   et la frequence emise reelle est :
%       f0 = sqrt(f_max * f_min) * sqrt(1 - (v/c)^2)
%
%   On selectionne f_max au debut (avant t1) et f_min a la fin (apres t1).

t1_estime = t_f_v(round(end/2));     % a affiner avec le spectrogramme

before = t_f_v < t1_estime - 0.1;
after  = t_f_v > t1_estime + 0.1;

f_max = median(maxk(f_t_lisse(before), max(3, round(0.1*sum(before)))));
f_min = median(mink(f_t_lisse(after) , max(3, round(0.1*sum(after )))));

v_estime = c * (f_max - f_min) / (f_max + f_min);
f0_estime = sqrt(f_max * f_min) * sqrt(1 - (v_estime/c)^2);

fprintf('\n--- Estimation Le Mans ---\n');
fprintf('f_max          : %.1f Hz\n', f_max);
fprintf('f_min          : %.1f Hz\n', f_min);
fprintf('f0 (estimee)   : %.1f Hz\n', f0_estime);
fprintf('Vitesse estimee : %.2f m/s   (%.1f km/h)\n', v_estime, v_estime*3.6);

%% --- 7. Verification par ajustement du modele Doppler ------------------
%   On suppose un passage perpendiculaire a une distance d (a estimer
%   par l'utilisateur si possible).
d_suppose = 10;          % a adapter (m) - hypothese pour le fit
modele = @(p, tt) p(1) * c ./ ( c + p(2)^2 * (tt - p(3)) ./ ...
                                sqrt(d_suppose^2 + p(2)^2*(tt - p(3)).^2) );

p0   = [f0_estime, v_estime, t1_estime];
opts = optimset('Display','off');
p_fit = lsqcurvefit(modele, p0, t_f_v, f_t_lisse, ...
                    [0,    0, 0], [5000, 150, max(t_f_v)], opts);

f0_fit = p_fit(1); v_fit = p_fit(2); t1_fit = p_fit(3);

f_modele = modele(p_fit, t_f_v);
figure('Name','Ajustement modele Le Mans');
plot(t_f_v, f_t_lisse, 'r-','LineWidth',1.2); hold on;
plot(t_f_v, f_modele , 'k--','LineWidth',1.5);
xlabel('t (s)'); ylabel('f (Hz)');
title(sprintf('Ajustement : f_0 = %.0f Hz, v = %.1f m/s (%.1f km/h)', ...
              f0_fit, v_fit, v_fit*3.6));
legend('Mesure','Modele'); grid on;

fprintf('\nFit non-lineaire (d suppose = %.1f m) :\n', d_suppose);
fprintf('  f0 = %.1f Hz\n  v  = %.2f m/s (%.1f km/h)\n  t1 = %.2f s\n', ...
        f0_fit, v_fit, v_fit*3.6, t1_fit);
