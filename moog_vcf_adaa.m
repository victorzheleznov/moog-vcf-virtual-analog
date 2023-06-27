% ------------------------------------------------------------------------- 
% Script for simulating nonlinear Moog VCF circuit
%
% The midpoint method is used to discretize the ODE system for the circuit. 
% The Newton-Raphson algorithm is implemented to solve discretized 
% equations.
%
% The regular digital filter model is compared with antialiased digital
% filter model [1]. Antiderivative antialiasing (ADAA) method is used 
% which replaces nonlinear function tanh() with its approximation using
% antiderivative function ln(cosh()).
%
% The script has two possible input signals:
% 1. Sine - to check filter distortion.
% 2. From a file - to apply filter to a recording.
%
% NB: there is no built-in resampling in a script so the sample rate of an 
% input file should match the simulation sample rate. Otherwise the input
% will be changed to sine wave.
%
% References:
% [1] Paschou, E., Esqueda Flores, F., Valimaki, V., & Mourjopoulos, J. 
%     (2017). Modeling and measuring a Moog voltage-controlled filter. In 
%     Proceedings of the Ninth Annual Conference of the Asia-Pacific Signal 
%     and Information Processing Association (pp. 1641 - 1647). IEEE. 
%     https://doi.org/10.1109/APSIPA.2017.8282295
%
% Author: Victor Zheleznov 
% Date: 20/02/2023
% -------------------------------------------------------------------------

clear all; close all; clc;

% define parameters
SR = 44100;       % sample rate [Hz]
Tf = 1;           % duration [sec]
f0 = 3.7e3;       % cutoff frequency [Hz]
r = 0.1;          % resonance (0 <= r <= 1)
tol_nr = 1e-8;    % tolerance value for Newton-Raphson algorithm
max_iter = 10;    % max iterations for Newton-Raphson algorithm
tol_adaa = 1e-10; % tolerance value for ADAA method

% filter input parameters
input = "sine"; % sine or file
fu = 500;       % input sine wave frequency [Hz]
Au = 100;       % input sine wave amplitude
file_path = "amen_break.wav";

% check allowed values
assert(((r >= 0) && (r <= 1)), "Resonance should be in [0,1] for stability!");
assert(isinf(cosh(Au)) == 0, 'Input signal amplitude is too high for cosh() function used in ADAA algorithm!');

% calculate derived parameters
w0 = 2*pi*f0;      % angular cutoff frequency [rad/sec]
Nf = floor(Tf*SR); % duration in samples
k = 1/SR;          % time step [sec]

% generate filter input
if strcmp(input, "sine") == 1
    u = Au*sin(2*pi*fu*(0:Nf-1)*k).';
elseif strcmp(input, "file") == 1
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
    error("Input should be set to sine or file!");
end

% initialise state, output and input vectors
x = zeros(4,1);
y = zeros(Nf,1);
xaa = zeros(4,1);
yaa = zeros(Nf,1);

% main loop
for n = 2:Nf
    % find current state
    x = midpoint_find_root(x, u(n), u(n-1), w0, r, k, tol_nr, max_iter, false, tol_adaa);    % regular midpoint method
    xaa = midpoint_find_root(xaa, u(n), u(n-1), w0, r, k, tol_nr, max_iter, true, tol_adaa); % midpoint method with ADAA
    % save output
    y(n) = x(4);
    yaa(n) = xaa(4);
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
    yaa_frame = yaa.*win;
    Yaa = fft(yaa_frame, NFFT);
    Yaa = Yaa(1:NFFT_2);

    % plot
    fig_fft = figure;
    fvec = (0:SR/NFFT:SR/2).';
    semilogx(fvec, 20*log10(abs(Y)./max(abs(Y))), 'b', fvec, 20*log10(abs(Yaa)./max(abs(Yaa))), 'r');
    xlim([10 SR/2]);
    ylim([-100 5]);
    xlabel('Frequency [Hz]', 'Interpreter', 'latex');
    ylabel('Magnitude [dB]', 'Interpreter', 'latex');
    title(sprintf('Digital filter measurement with a sine wave input signal with amplitude $A = %.1f$ and frequency $f = %.1f$ Hz', Au, fu), 'Interpreter', 'latex');
    legend({'Regular midpoint method', 'Midpoint method with ADAA'}, 'Interpreter', 'latex');
end

% plot spectogram for filtered audio file and save the result
if strcmp(input, "file")
    myspec(yaa, SR, 2048, 0.75);
    [~,file_name] = fileparts(file_path);
    audiowrite(append(file_name, '_moog_vcf_adaa', '.wav'), yaa./max(abs(yaa)), SR);
end

%% FUNCTIONS
% calculate implicit form of the Moog VCF circuit equations discretized
% using midpoint tule
% input:
%   x - current state vector;
%   xd - delayed state vector;
%   u - current input value;
%   ud - delayed input value;
%   w0 - angular cutoff frequency [rad/sec];
%   r - resonance (0 <= r <= 1);
%   k - sampling time step [sec];
%   USE_ADAA - flag to use antiderivative antialiasing method;
%   tol - tolerance value for antiderivative antialiasing method.
% output:
%   G - implicit vector function;
%   DG - Jacobian matrix for G.
function [G, DG] = midpoint_calculate_implicit_form(x, xd, u, ud, w0, r, k, USE_ADAA, tol)
    % calculate derived parameters
    S0 = 4*r*(x(4)+xd(4))/2 + (u+ud)/2;
    J0 = (k*w0/2)*(1 - tanh(S0)^2);
    S = (x+xd)/2;
    J = (k*w0/2)*(1 - tanh(S).^2);
    if USE_ADAA == false
        % calculate matrices
        G = x - xd - k*w0*[-tanh(S(1))-tanh(S0); -tanh(S(2))+tanh(S(1)); -tanh(S(3))+tanh(S(2)); -tanh(S(4))+tanh(S(3))];
        DG = midpoint_calculate_jacobian(J0,J,r);
    else
        % calculate derived parameters
        I0 = log(cosh(4*r*x(4)+u));
        I0d = log(cosh(4*r*xd(4)+ud));
        d0 = 4*r*x(4)+u - (4*r*xd(4)+ud);
        Z0 = (k*w0/d0^2)*(d0*tanh(4*r*x(4)+u) - (I0-I0d));
        I = log(cosh(x));
        Id = log(cosh(xd));
        d = x - xd;
        Z = (k*w0)./(d.^2).*(d.*tanh(x) - (I-Id));
        % calculate antialiased tanh()
        if abs(d0) > tol
            tanh_aa_0 = (I0-I0d)/d0;
        else
            tanh_aa_0 = tanh(S0);
            Z0 = J0;
        end
        tanh_aa = zeros(size(I));
        for i = 1:length(x)
            if abs(d(i)) > tol
                tanh_aa(i) = (I(i)-Id(i))/d(i);
            else
                tanh_aa(i) = tanh(S(i));
                Z(i) = J(i);
            end
        end
        % calculate matrices
        G = x - xd - k*w0*[-tanh_aa(1)-tanh_aa_0; -tanh_aa(2)+tanh_aa(1); -tanh_aa(3)+tanh_aa(2); -tanh_aa(4)+tanh_aa(3)];
        DG = midpoint_calculate_jacobian(Z0,Z,r);
    end
end

% calculate Jacobian matrix for implicit Moog VCF circuit equations
% discretized by midpoint rule
function DG = midpoint_calculate_jacobian(J0,J,r)
    DG = [ 1+J(1),  0     ,  0     , 4*r*J0;
          -J(1)  ,  1+J(2),  0     , 0     ;
           0     , -J(2)  ,  1+J(3), 0     ;
           0     ,  0     , -J(3)  , 1+J(4)];
end

% calculate next filter sample using the Newton-Raphson root finding 
% algorithm for implicit Moog VCF circuit equations discretized by midpoint
% rule
% input:
%   xd - delayed state vector;
%   u - current input value;
%   ud - delayed input value;
%   w0 - angular cutoff frequency [rad/sec];
%   r - resonance (0 <= r <= 1);
%   k - sampling time step [sec];
%   tol_nr - tolerance value for the Newton-Raphson algorithm;
%   max_iter - max iterations for Newton-Raphson algorithm;
%   USE_ADAA - flag to use antiderivative antialiasing method;
%   tol_adaa - tolerance value for antiderivative antialiasing method.
% output:
%   x - current state vector.
function x = midpoint_find_root(xd, u, ud, w0, r, k, tol_nr, max_iter, USE_ADAA, tol_adaa)
    % initialise
    iter = 0;
    step = ones(size(xd));
    x = rand(size(xd));
    % main loop
    while (norm(step) > tol_nr) && (iter < max_iter)
        % obtain implicit function and its Jacobian matrix
        [G,DG] = midpoint_calculate_implicit_form(x, xd, u, ud, w0, r, k, USE_ADAA, tol_adaa);
        % apply the Newton-Raphson algorithm
        step = DG\G;
        x = x - step;
        iter = iter + 1;
    end
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