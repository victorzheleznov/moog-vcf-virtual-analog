% ------------------------------------------------------------------------- 
% Script for simulating nonlinear Moog VCF circuit
%
% The nonlinear Moog VCF equations are transformed to the form of Port 
% Hamiltonian Systems [1]. These equations are disretized to obtain 
% explicit numerical methods of first and second order of accuracy [2].
%
% The script has three possible input signals:
% 1. None - to test the energy balance and zero-input stability.
% 2. Sine - to check filter distortion.
% 3. From a file - to apply filter to a recording.
%
% NB: there is no built-in resampling in a script so the sample rate of an 
% input file should match the simulation sample rate. Otherwise the input
% will be changed to sine wave.
%
% References:
% [1] Danish, M., Bilbao, S., & Ducceschi, M. (2021). Applications of Port 
%     Hamiltonian Methods to Non-Iterative Stable Simulations of the KORG35 
%     and MOOG 4-Pole VCF. 2021 24th International Conference on Digital 
%     Audio Effects (DAFx), 33-40.
% [2] Lopes, N., Helie, T., & Falaize, A. (2015). Explicit second-order 
%     accurate method for the passive guaranteed simulation of 
%     port-Hamiltonian systems. IFAC-PapersOnLine, 48, 223-228.
%     https://doi.org/10.1016/j.ifacol.2015.10.243
%
% Author: Victor Zheleznov 
% Date: 23/02/2023
% -------------------------------------------------------------------------

clear all; close all; clc;

% define parameters
SR = 44100;   % sample rate [Hz]
Tf = 1;       % simulation duration [sec]
f0 = 3.7e3;   % cutoff frequency [Hz]
r = 0.1;      % resonance (0 < r <= 1)
tol = 1e-6;   % tolerance value for Taylor expansions in PHS matrices
accuracy = 2; % numerical method accuracy order (first or second) [2]
p = 0.5;      % parameter for two-stage method [2]

% filter input parameters
input = "sine";    % "none", "sine" or "file"
x0 = [1; 1; 1; 1]; % initial state (is used for "none" input mode)
fu = 500;          % input sine wave frequency [Hz]
Au = 100;          % input sine wave amplitude
file_path = "amen_break.wav";

% check allowed values
assert(((r > 0) && (r <= 1)), "Resonance should be in (0,1] for stability!");

% calculate derived parameters
w0 = 2*pi*f0;        % angular cutoff frequency [rad/sec]
Nf = floor(Tf*SR);   % duration in samples
k = 1/SR;            % time step [sec]
a = sqrt(2)*r^(1/4); % transformed resonance
d = max(1,a);        % parameter for Lyapunov function stabilty
q = 1/(2*(1-p));     % parameter for Runge-Kutta point

% generate filter input
if strcmp(input, "none") == 1
    x = x0;
    u = zeros(Nf,1);
elseif strcmp(input, "sine") == 1
    x = zeros(4,1);
    u = Au*sin(2*pi*fu*(0:Nf-1)*k).';
elseif strcmp(input, "file") == 1
    x = zeros(4,1);
    [u, uSR] = audioread(file_path);
    if size(u,2) == 2
        u = mean(u,2);
    end
    if uSR == SR
        Nf = length(u);
    else
        warning("Input file has a different sample rate! Changed input to sine");
        input = "sine";
        u = Au*sin(2*pi*fu*(0:Nf-1)*k).';
    end
else
    error("Input should be set to none, sine or file!");
end

% initialise arrays
y = zeros(Nf,1); % filter output
H = zeros(Nf,1); % stored energy
D = zeros(Nf,1); % dissipated energy

% set initial values
w = x_to_w(x, d);
z = w_to_z(w, d, a);
y(1) = x(4);
H(1) = 0.5*(z.')*z;

% define identity matrix
I = eye(length(z));

% main loop
for n = 2:Nf
    % calculate state step
    [S,G] = calc_PHS(z, u(n-1), d, a, w0, tol);
    if accuracy == 1
        dz = k*((I-(k/2)*S)\(S*z + G));
    elseif accuracy == 2
        dz1 = k*((I-(k/2)*p*S)\(S*z + G));
        zp = z + p*dz1;
        zq = z + q*dz1;
        [Sq,Gq] = calc_PHS(zq, u(n-1), d, a, w0, tol);
        dz2 = k*((I-(k/2)*(1-p)*Sq)\(Sq*zp + Gq));
        dz = p*dz1 + (1-p)*dz2;
    end
    % calculate dissipated energy
    if accuracy == 1
        D(n) = D(n-1) - k*((z+0.5*dz).')*S*(z+0.5*dz);
    elseif accuracy == 2
        D(n) = D(n-1) - p*k*((z+0.5*p*dz1).')*S*(z+0.5*p*dz1) - (1-p)*k*((zp+0.5*(1-p)*dz2).')*Sq*(zp+0.5*(1-p)*dz2);
    end
    % calculate current state
    z = z + dz;
    % calculate output
    w = z_to_w(z, d, a);
    x = w_to_x(w, d);
    y(n) = x(4);
    % calculate stored energy
    H(n) = 0.5*(z.')*z;
end

% plot filter measurement for sine input
if strcmp(input, "sine") == 1
    % calculate FFT parameters
    NFFT = 2^(ceil(log(Nf)/log(2)));         % fft size
    NFFT_2 = NFFT / 2 + 1;
    L = Nf;                                  % window length
    win = 0.5*(1 - cos(2*pi.*(0:L-1)./L)).'; % hanning window

    % calculate FFT
    y_frame = y.*win;
    Y = fft(y_frame, NFFT);
    Y = Y(1:NFFT_2);

    % plot
    fig_fft = figure;
    fvec = (0:SR/NFFT:SR/2).';
    semilogx(fvec, 20*log10(abs(Y)./max(abs(Y))), 'b');
    xlim([10 SR/2]);
    ylim([-100 5]);
    xlabel('Frequency [Hz]', 'Interpreter', 'latex');
    ylabel('Magnitude [dB]', 'Interpreter', 'latex');
    title(sprintf('Digital filter measurement with a sine wave input signal with amplitude $A = %.1f$ and frequency $f = %.1f$ Hz', Au, fu), 'Interpreter', 'latex');
end

% plot energy graph to check zero-input stability
if strcmp(input, "none") == 1
    fig_h = figure;
    t = (0:Nf-1)*k;
    E = H+D;

    subplot(2,1,1);
    plot(t, H, 'b', t, D, 'r', t, E, 'g');
    xlabel('Time [sec]', 'Interpreter', 'latex');
    ylabel('Energy', 'Interpreter', 'latex');
    legend({'Stored energy', 'Dissipated energy', 'Total energy'}, 'Interpreter', 'latex');
    title('Energy variation', 'Interpreter', 'latex');

    subplot(2,1,2)
    plot(t, (E-E(1))/E(1), 'b');
    xlabel('Time [sec]', 'Interpreter', 'latex');
    ylabel('$\frac{E^n - E^0}{E^0}$', 'Interpreter', 'latex');
    title('Relative total energy difference', 'Interpreter', 'latex');
end

% plot spectogram for filtered audio file and save the result
if strcmp(input, "file")
    myspec(y, SR, 2048, 0.75);
    [~,file_name] = fileparts(file_path);
    audiowrite(append(file_name, '_moog_vcf_phs', '.wav'), y./max(abs(y)), SR);
end

%% FUNCTIONS
% transform from x to w [1]
% input:
%   x - state vector in x domain;
%   d - transformation parameter.
% output:
%   w - state vector in w domain.
function w = x_to_w(x, d)
    w = diag([1,d,d^2,d^3])*x;
end

% transform from w to x [1]
% input:
%   w - state vector in w domain;
%   d - transformation parameter.
% output:
%   x - state vector in x domain.
function x = w_to_x(w, d)
    x = diag([1,1/d,1/d^2,1/d^3])*w;
end

% transform from w to z [1]
% input:
%   w - state vector in w domain;
%   d - transformation parameter from x to w;
%   a - transformed resonance.
% output:
%   z - state vector in z domain.
function z = w_to_z(w, d, a)
    z = sqrt(2)*[sign(w(1))*sqrt(log(cosh(w(1))));...
                 sign(w(2))*d*sqrt(log(cosh(w(2)/d)));...
                 sign(w(3))*d^2*sqrt(log(cosh(w(3)/d^2)));...
                 sign(w(4))*d/a^2*sqrt(log(cosh(a^4*w(4)/d^3)))];
end

% transform from z to w [1]
% input:
%   z - state vector in z domain;
%   d - transformation parameter from x to w;
%   a - transformed resonance.
% output:
%   w - state vector in w domain.
function w = z_to_w(z, d, a)
    w = [sign(z(1))*acosh(exp(z(1)^2/2));...
         sign(z(2))*d*acosh(exp(z(2)^2/(2*d^2)));...
         sign(z(3))*d^2*acosh(exp(z(3)^2/(2*d^4)));...
         sign(z(4))*d^2/a^4*acosh(exp(a^4*z(4)^2/(2*d^2)))];
end

% calculate matrices for PHS representation of Moog VCF [1]
% input:
%   z - current transformed state vector;
%   u - current input value;
%   d - transformation parameter from x to w;
%   a - transformed resonance;
%   w0 - angular cutoff frequency [rad/sec];
%   tol - tolerance value for Taylor expansions.
% output:
%   S - power exchanges matrix in z domain;
%   G - input matrix in z domain.
function [S,G] = calc_PHS(z, u, d, a, w0, tol)
    % Jacobian matrix for state change
    J = diag([sigma(z(1),tol); sigma(z(2)/d,tol); sigma(z(3)/d^2,tol); a^2/d^2*sigma(a^2*z(4)/d,tol)]);
    % PHS matrices
    S = w0*[-1,  0,  0, -d;
             d, -1,  0,  0;
             0,  d, -1,  0;
             0,  0,  d, -g(z(4),d,a,tol)];
    w = z_to_w(z, d, a);
    G = w0*[gamma(a^4*w(4)/d^3,u); 0; 0; 0];
    % convert to new variables
    S = J.'*S*J;
    G = J.'*G;
end

% sigma function [1]
% input:
%   x - input value;
%   tol - tolerance value for Taylor expansion.
% output:
%   y - function value.
function y = sigma(x, tol)
    if abs(x) > tol
        y = sqrt((1-exp(-x^2))/x^2);
    else
        y = 1 - x^2/4;
    end
end

% g function [1] (article has a typo in Taylor expansion - no argument in numerator)
% input:
%   x - input value;
%   d - transformation parameter from x to w;
%   a - transformed resonance;
%   tol - tolerance value for Taylor expansion.
% output:
%   y - function value.
function y = g(x, d, a, tol)
    arg = a^4*x^2/d^2;
    if abs(arg) > tol
        y = d^4*tanh(acosh(exp(arg/2))/a^4)/sqrt(1-exp(-arg));
    else
        y = (d^4/a^4)*(1 + (a^8-4)/(12*a^8)*arg)/(1 - 0.25*arg);
    end
end

% gamma function [1]
% input:
%   x, u - input scalars.
% output:
%   y - function value.
function y = gamma(x, u)
    y = -tanh(u)*(1-tanh(x)^2)/(1+tanh(u)*tanh(x));
end

% create a spectogram plot of an input signal
% input:
%   x - mono input signal;
%   Fs - sampling frequency [Hz];
%   N - frame length;
%   O - overlap factor (between 0 and 1).
function [] = myspec(x, Fs, N, O)
    % find hop size
    HA = round(N - O*N);

    % generate window
    win = 0.5*(1 - cos(2*pi.*(0:N-1)./N)).';

    % calculate number of frames
    L = length(x);
    NF = ceil(L/HA);
    x = [x; zeros((NF-1)*HA+N-L,1)];
    
    % STFT size
    NFFT = 2^(ceil(log(N)/log(2))); % next power of 2
    NFFT_2 = NFFT / 2 + 1;

    % calculate STFT
    STFT = zeros(NFFT_2, NF);
    for m = 0:NF-1
        x_frame = win.*x((1:N).'+m*HA);
        X = fft(x_frame, NFFT);
        STFT(:,m+1) = X(1:NFFT_2);
    end
    
    % plot spectogram
    fig_spec = figure;
    t = ((0:NF-1).*HA/Fs).';
    freq = (0:Fs/NFFT:Fs/2).';
    STFT_dB = 20*log10(abs(STFT));
    max_dB = max(max(STFT_dB));
    imagesc(t, freq, STFT_dB, 'CDataMapping', 'scaled');
    c = colorbar;
    c.Label.String = 'dB';
    colormap hot
    caxis([max_dB-60, max_dB]);
    xlim([0 t(end)]);
    ylim([0 freq(end)]);
    ax_spec = fig_spec.CurrentAxes;
    set(ax_spec, 'YDir', 'normal');
    set(ax_spec, 'YTick', 0:1000:Fs/2);
    set(ax_spec, 'YTickLabel', 0:1000:Fs/2);
    xlabel('Time [s]', 'interpreter', 'latex');
    ylabel('Frequency [Hz]', 'interpreter', 'latex');
    title_str = sprintf("Spectogram with frame length = $%d$ ms and overlap factor = $%d$\\%%", floor((N/Fs)*1e3), O*1e2);
    title(title_str, 'interpreter', 'latex');
end