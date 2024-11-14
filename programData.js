`

const programData = {
    subjects: {
        DSP: [
            {
                header: "DSP Program Placeholder",
                question: "DSP Program Question Placeholder",
                code: "% DSP Program Code Placeholder"
            },
            // Additional DSP programs can be added here
            /*
            {
                header: "Additional DSP Program Placeholder",
                question: "Additional DSP Program Question Placeholder",
                code: "% Additional DSP Program Code Placeholder"
            }
            */
        ],
        DC: [
            {
                header: "DC Program Placeholder",
                question: "DC Program Question Placeholder",
                code: "% DC Program Code Placeholder"
            },
            // Additional DC programs can be added here
            /*
            {
                header: "Additional DC Program Placeholder",
                question: "Additional DC Program Question Placeholder",
                code: "% Additional DC Program Code Placeholder"
            }
            */
        ],
        
        // Future placeholders for another subject set (commented out)
        /*
        AnotherSubject: [
            {
                header: "Another Subject Program Placeholder",
                question: "Another Subject Program Question Placeholder",
                code: "% Another Subject Program Code Placeholder"
            },
            // Additional programs for AnotherSubject can be added here
            {
                header: "Additional AnotherSubject Program Placeholder",
                question: "Additional AnotherSubject Program Question Placeholder",
                code: "% Additional AnotherSubject Program Code Placeholder"
            }
        ]
        */
    }
}

`
    
const programData = {
    subjects: {
        DSP: [
            {
                header: "DSP Program Placeholder",
                question: "DFT and IDFT Linear  Convolution",
                code: 
                `
clc;
clear all;
close all;
x=input('enter the first input sequence=');
h=input('enter the second input sequence=');
l=length(x);
m=length(h);
N=max(l,m);
Xk=fft(x,N);
Hk=fft(h,N);
Yk=Xk.*Hk;
y=ifft(Yk,N);
disp('circuler convoluted output using DFT and IDFT method');
disp(y);
subplot(3,1,1);
stem(x);
title('the first sequence');
xlabel('time');
ylabel('amplitude');
subplot(3,1,2);
stem(h);
title('the second sequence');
xlabel('time');
ylabel('amplitude');
subplot(3,1,3);
stem(y);
title('the circuler convoluted sequence');
xlabel('time');
ylabel('amplitude');
                `
            },
            {
                header: "DSP Program Placeholder",
                question: "DFT and IDFT Circular Convolution",
 code: 
                `
clc;
clear all;
close all;
x=input('enter the first input sequence=');
h=input('enter the second input sequence=');
l=length(x);
m=length(h);
N=l+m-1;
Xk=fft(x,N);
Hk=fft(h,N);
Yk=Xk.*Hk;
y=ifft(Yk,N);
disp('linear convoluted output using DFT and IDFT method');
disp(y);
subplot(3,1,1);
stem(x);
title('the first sequence');
xlabel('time');
ylabel('amplitude');
subplot(3,1,2);
stem(h);
title('the second sequence');
xlabel('time');
ylabel('amplitude');
subplot(3,1,3);
stem(y);
title('the linear convoluted sequence');
xlabel('time');
ylabel('amplitude');

                
                `            },
            {
                header: "DSP Program Placeholder",
                question: "Linearity property",
 code: 
                `
clc;
clear all;
close all;
x1=input('enter the first input sequence=');
x2=input('enter the second input sequence=');
a1=input('enter the constant a1=');
a2=input('enter the constant a2=');
l1=length(x1);
l2=length(x2);
N=max(l1,l2);
x1n=[x1,zeros(1,N-l1)];
x2n=[x2,zeros(1,N-l2)];
y=a1*x1n+a2*x2n;
yk=fft(y,N);
disp('output sequence y(k) is');
disp(yk);
x1k=fft(x1,N);
x2k=fft(x2,N);
yv=a1*x1k+a2*x2k;
if(yk==yv)
    disp('linearity property is satisfied');
    else
      disp('linearity property is not satisfied');  
end;
                
                `            },
            {
                header: "DSP Program Placeholder",
                question: "Circular time shift property",
 code: 
                `
 clc;
clear all;
close all;
x=input('enter the first input sequence=');
m=input('enter the number of shifts');
N=length(x);
Xs=circshift(x,[0,m]);
y=fft(Xs,N);
xk=fft(x,N);
for K=0:N-1
    w(K+1)=exp((-j*2*pi*K*m)/N);
end;
Yv=w.*xk;
disp(y);
disp(Yv);
if(floor(abs(y)))==(floor(abs(Yv)))
    disp('circular time shift property is satisfied');
    else
      disp('circular time shift property is not satisfied');  
end;
               
                `            },
            {
                header: "DSP Program Placeholder",
                question: "Circular frequency shift property",
 code: 
                `
clc;
clear all;
close all;
x=input('enter the first input sequence=');
l=input('enter the number of shifts');
N=length(x);
xk=fft(x,N);
yv=circshift(xk,[0,l]);
for n=0:N-1
    w(n+1)=exp((j*2*pi*n*l)/N);
end;
y=w.*x;
yk=fft(y);
disp(yk);
disp(yv);
if(floor(abs(yk)))==(floor(abs(yv)))
    disp('circular frequency shift property is satisfied');
    else
      disp('circular frequency shift property is not satisfied');  
end;
                
                `            },
            {
                header: "DSP Program Placeholder",
                question: "Analyse causal system",
 code: 
                `
% Define the transfer function H(z)
syms z;
H_z = 1 / (1 - 0.9*z^(-1)); % Transfer function in terms of z
% Display the transfer function
disp('Transfer function H(z) is:');
pretty(H_z);
% Convert to a rational transfer function (polynomial in z)
H_z_poly = simplify(H_z);
H_z_poly = collect(H_z_poly, z);
% Plot the frequency response of H(z)
f = linspace(-pi, pi, 1000); % Frequency range from -pi to pi
H_w = @(w) 1 ./ (1 - 0.9 * exp(-1i * w)); % Frequency response H(w)
% Calculate the magnitude and phase of H(z) over the frequency range
mag_H = abs(H_w(f)); % Magnitude of the transfer function
phase_H = angle(H_w(f)); % Phase of the transfer function
% Plot magnitude and phase response
figure;
subplot(2,1,1);
plot(f, mag_H);
title('Magnitude Response of H(z)');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
subplot(2,1,2);
plot(f, phase_H);
title('Phase Response of H(z)');
xlabel('Frequency (rad/sample)');
ylabel('Phase (radians)');
% Plot the pole-zero plot
figure;
pzplot(H);
title('Pole-Zero Plot of H(z)');
% Define the number of samples
N = 10; % Number of samples to display (can be adjusted)
% Define the time index n (0 to N-1)
n = 0:N-1;
% Compute the impulse response h(n) = 0.9^n for n >= 0
h_n = (0.9).^n; % Impulse response
% Display the impulse response values
disp('Impulse Response h(n):');
disp(h_n);
% Plot the impulse response
figure;
stem(n, h_n, 'filled', 'MarkerSize', 6);
title('Impulse Response h(n) = 0.9^n');
xlabel('n');
ylabel('h(n)');
grid on;                
                `            },
            {
                            header: "DSP Program Placeholder",
                question: "Analyse causal system2",
 code: 
                `
% Define the transfer function H(z)

num = [1]; % Numerator coefficients (z)
den = [1 -0.9]; % Denominator coefficients (z - 0.9)

% Create the transfer function
H = tf(num, den, -1); % -1 indicates discrete time
disp('seros and poles Located at')
[zz,pp] = tf2zp(num,den)

% a) Pole-Zero Plot
figure;
pzmap(H);
title('Pole-Zero Plot of H(z)');
grid on;

% b) Frequency Response
omega = linspace(-pi, pi, 1024); % Frequency range
H_freq = freqz(num, den, omega); % Frequency response

% Magnitude and Phase
magnitude = abs(H_freq);
phase = angle(H_freq);

% Plot Magnitude Response
figure;
subplot(2, 1, 1);
plot(omega, magnitude);
title('Magnitude Response');
xlabel('frequency in radians');
ylabel('Magnitude');

% Plot Phase Response
subplot(2, 1, 2);
plot(omega, phase);
title('Phase Response');
xlabel('frequency in radians');
ylabel('Phase values');


% c) Impulse Response
n = 0:20; % Time index
h = (0.9).^n; % Impulse response for n >= 0
h(1) = 1; % h(0) = 1 for the impulse response
disp('Samples of impulse response')
disp(h);
figure;
stem(n, h);
title('Impulse Response h(n)');
xlabel('n');
ylabel('Amplitude');
           
                `            },
            {
                header: "DSP Program Placeholder",
                question: "dit_radix2_fft",
 code: 
                `

% a)This part of the program is for creating “dit_radix2_fft” function 
%Note:student must type the following code and save as dit_radix2_fft

function X = dit_radix2_fft(x)
    % DIT Radix-2 FFT Implementation
    % Input:
    %   x - Input signal (must be a power of 2 in length)
    % Output:
    %   X - FFT of the input signal

    % Check if the length of x is a power of 2
    N = length(x);
    if mod(log2(N), 1) ~= 0
        error('Input length must be a power of 2');
    end

    % Bit-reverse the input
    x = bit_reverse(x);

    % Initialize the output
    X = x;

    % Number of stages
    stages = log2(N);

    % FFT computation
    for stage = 1:stages
        % Number of points in each FFT stage
        num_points = 2^stage;
        half_points = num_points / 2;

        % Twiddle factors
        W = exp(-2 * pi * 1i * (0:half_points-1) / num_points);

        for k = 0:N/num_points-1
            for j = 0:half_points-1
                % Indices for the butterfly operation
                idx1 = k * num_points + j + 1; % MATLAB is 1-indexed
                idx2 = idx1 + half_points;

                % Butterfly operation
                temp = X(idx2) * W(j + 1);
                X(idx2) = X(idx1) - temp;
                X(idx1) = X(idx1) + temp;
            end
        end
    end
end

function x_bitreversed = bit_reverse(x)
    % Bit-reverse the input array
    N = length(x);
    n = log2(N);
    x_bitreversed = zeros(size(x));

    for k = 0:N-1
        % Bit-reverse the index
        rev_idx = bin2dec(fliplr(dec2bin(k, n))) + 1; % MATLAB is 1-indexed
        x_bitreversed(rev_idx) = x(k + 1);
    end
end



% b): Example program to Develop decimation in time radix-2 FFT algorithm 

clc;
clear all;
close all;
N = 4; % Length of the input signal (must be a power of 2)
x = [1 2 3 4]; % Example input signal
X = dit_radix2_fft(x); % Compute FFT
disp('Input Signal:');
disp(x);
disp('FFT of the Input Signal:');
disp(X);
                
                `            },
            {
                header: "DSP Program Placeholder",
                question: "digital low pass FIR filter using Rectangular /Bartlett/Hamming/Hanning window",
 code: 
                `
  clc;
clear all;
close all;
rp=input('enter the passband ripple');
rs=input('enter the stopband ripple');
fp=input('enter the passbandfreq');
fs=input('enter the stopbandfreq');
f=input('enter the sampling freq');
wp=2*fp/f;
ws=2*fs/f;
num=-20*log10(sqrt(rp*rs))-13;
den=14.6*(fs-fp)/f;
n=ceil(num/den);
n1=n+1;
if(rem(n,2)~=0)
 n1=n;
 n=n-1;
end;
y=boxcar(n1); / y=bartlett(n1); / y=hanning(n1); / y=hamming(n1);
b=fir1(n,wp,'low',y);
[h,q]=freqz(b,1,256);
m=20*log10(abs(h));
subplot(2,2,1);
plot(q/pi,m);
xlabel('normalisedfreq');
ylabel('gain in dB');
title('low pass filter using rectangular window');
system output
enter the passband ripple=0.02
enter the stopband ripple=0.01
enter the passbandfreq=1200
enter the stopbandfreq=1700
enter the sampling freq=9000              
                `            },
            {
                header: "DSP Program Placeholder",
                question: "digital high pass FIR filter using Rectangular /Bartlett/Hamming/Hanning window",
 code: 
                `
clc;
clearall;
closeall;
rp=input('enter the passband ripple');
rs=input('enter the stopband ripple');
fp=input('enter the passbandfreq');
fs=input('enter the stopbandfreq');
f=input('enter the sampling freq');
wp=2*fp/f;
ws=2*fs/f;
num=-20*log10(sqrt(rp*rs))-13;
den=14.6*(fs-fp)/f;
n=ceil(num/den);
n1=n+1;
if(rem(n,2)~=0)
 n1=n;
 n=n-1;
end;
y=boxcar(n1); / y=bartlett(n1); / y=hanning(n1); / y=hamming(n1);
b=fir1(n,wp,'high',y);
[h,q]=freqz(b,1,256);
m=20*log10(abs(h));
subplot(2,2,2);
plot(q/pi,m);
xlabel('normalisedfreq');
ylabel('gain in dB');
title('high pass filter using rectangular window');
system output
enter the passband ripple=0.02
enter the stopband ripple=0.01
enter the passbandfreq=1200
enter the stopbandfreq=1700
enter the sampling freq=9000
                
                `            },
            {
                header: "DSP Program Placeholder",
                question: "digital IIR Butterworth low pass filter",
 code: 
                `
 clc;
clear all;
close all;
rp=input('enter the pass band ripple=');
rs=input('enter the stop band ripple=');
wp=input('enter the pass band frequency=');
ws=input('enter the stop band frequency=');
fs=input('enter the sampling frequency=');
w1=2*wp/fs;
 w2=2*ws/fs;
 [n,wn]=buttord(w1,w2,rp,rs);
 [b,a]=butter(n,wn,'low');
disp('the order of lpf');
disp(n);
disp('the cut off freq of lpf');
disp(wn);
w=0:0.01:pi;
[h]=freqz(b,a,w);
mag=20*log10(abs(h));
ang=angle(h);
subplot(2,1,1);
plot(w/pi,mag);
xlabel('normalized freq');
ylabel('magnitude');
subplot(2,1,2);
plot(w/pi,ang);
xlabel('normalised freq');
ylabel('angle');
output:
enter the pass band ripple=3
enter the stop band ripple=60
enter the pass band frequency=150
enter the stop band frequency=300
enter the sampling frequency=1500               
                `            },
            {
                header: "DSP Program Placeholder",
                question: "digital IIR Butterworth high pass filter",
 code: 
                `
clc;
clear all;
close all;
rp=input('enter the pass band ripple=');
rs=input('enter the stop band ripple=');
wp=input('enter the pass band frequency=');
ws=input('enter the stop band frequency=');
fs=input('enter the sampling frequency=');
 w1=2*wp/fs;
 w2=2*ws/fs;
 [n,wn]=buttord(w1,w2,rp,rs);
 [b,a]=butter(n,wn,'high');
disp('the order of hpf');
disp(n);
disp('the cut off freq of hpf');
disp(wn);
 w=0:0.01:pi;
 [h]=freqz(b,a,w);
mag=20*log10(abs(h));
ang=angle(h);
subplot(2,1,1);
plot(w/pi,mag);
xlabel('normalized freq');
ylabel('magnitude');
subplot(2,1,2);
plot(w/pi,ang);
xlabel('normalisedfreq');
ylabel('angle');
output:
enter the pass band ripple=3
enter the stop band ripple=60
enter the pass band frequency=150
enter the stop band frequency=300
enter the sampling frequency=1500                
                `            },
            // Additional DSP programs can be added here
            /*
            {
                header: "Additional DSP Program Placeholder",
                question: "Additional DSP Program Question Placeholder",
                code: 
`% Additional DSP Program Code Placeholder`
            },
            */
        ],
        DC: [
            {
                header: "Prg 4:Gram-Schmidt Orthogonalization",
                question: "To find orthogonal basis vectors for the given set of vectors and plot the orthonormal vectors.",
                code: 
`clc; close all; clear all;

% Define the set of input vectors (3x3 matrix, each column is a vector)
V = [1 1 0; 1 0 1; 0 1 1]'; % Linearly Independent vectors
%V = rand(3, 3); % Uncomment for random vectors
%V = [1 2 3; 1 5 2; 2 4 6]'; % Uncomment for linearly dependent vectors

% Number of vectors (columns of the matrix)
n = size(V, 2); % The number of input vectors

% Initialize the matrix for storing orthogonal vectors
U = zeros(size(V)); 

% Gram-Schmidt Process: Convert linearly independent vectors to orthogonal vectors
for i = 1:n
    % Start with the original vector
    U(:, i) = V(:, i);

    % Subtract projections of all previous orthogonal vectors from the current vector
    for j = 1:i-1
        % Calculate the projection of V(:, i) onto U(:, j) and subtract it
        U(:, i) = U(:, i) - (dot(U(:, j), V(:, i)) / dot(U(:, j), U(:, j))) * U(:, j);
    end
end

% Normalize the orthogonal vectors to make them orthonormal
E = zeros(size(U)); % Initialize the matrix for orthonormal vectors
for i = 1:n
    % Normalize each vector (divide by its norm)
    E(:, i) = U(:, i) / norm(U(:, i));
end

% Display results for the original, orthogonal, and orthonormal vectors
disp('Original Vectors (V):');
disp(V);
disp('Orthogonal Vectors (U):');
disp(U);
disp('Orthonormal Vectors (E):');
disp(E);

% Plotting the original vectors
figure; % Create a new figure
hold on; % Retain current plot so we can overlay other vectors
grid on; % Enable grid

% Plot each original vector as a 3D arrow
quiver3(0, 0, 0, V(1, 1), V(2, 1), V(3, 1), 'r', 'LineWidth', 2); % Vector V1 (Red)
quiver3(0, 0, 0, V(1, 2), V(2, 2), V(3, 2), 'g', 'LineWidth', 2); % Vector V2 (Green)
quiver3(0, 0, 0, V(1, 3), V(2, 3), V(3, 3), 'b', 'LineWidth', 2); % Vector V3 (Blue)

% Plot the orthonormal vectors as dashed arrows
quiver3(0, 0, 0, E(1, 1), E(2, 1), E(3, 1), 'r--', 'LineWidth', 2); % Vector E1 (Red Dashed)
quiver3(0, 0, 0, E(1, 2), E(2, 2), E(3, 2), 'g--', 'LineWidth', 2); % Vector E2 (Green Dashed)
quiver3(0, 0, 0, E(1, 3), E(2, 3), E(3, 3), 'b--', 'LineWidth', 2); % Vector E3 (Blue Dashed)

% Setting up the plot labels, title, and legend
xlabel('X'); % X-axis label
ylabel('Y'); % Y-axis label
zlabel('Z'); % Z-axis label
title('Gram-Schmidt Orthonormalization'); % Plot title
legend({'V1', 'V2', 'V3', 'E1', 'E2', 'E3'}, 'Location', 'Best'); % Legend

% Make sure the axes are equally scaled
axis equal;

% Release the plot hold so further plotting won't affect this figure
hold off;`
            },

                {
                header: "Prg 5:QPSK",
                question: "Modulation and Demodulation oF Quadrature Phase Shift Keying",
                code: 
`clc;
clear all;
close all;

% Define binary data to transmit
data = [0 1 0 1 1 1 0 0 1 1]; 

% Plot the original data before transmission
figure(1)
stem(data, 'linewidth', 3), grid on;
title('Information before Transmitting');
axis([0 11 0 1.5]);

% Convert data to NZR format (0 -> -1, 1 -> 1)
data_NZR = 2 * data - 1; 

% Serial-to-Parallel (S/P) conversion for QPSK
s_p_data = reshape(data_NZR, 2, length(data) / 2); 

% Transmission parameters
br = 10^6; % Bit rate (1 Mbps)
f = br; % Carrier frequency
T = 1 / br; % Bit duration
t = T / 99 : T / 99 : T; % Time vector for one bit

% XXXXXXXXXXXXXXXXXXXXXXXX QPSK Modulation XXXXXXXXXXXXXXXXXXXXXXXXXXXX
y = []; % Initialize modulated signal
y_in = []; % In-phase signal
y_qd = []; % Quadrature signal

% Modulate each pair of data
for i = 1 : length(data) / 2
    y1 = s_p_data(1, i) * cos(2 * pi * f * t); % In-phase component
    y2 = s_p_data(2, i) * sin(2 * pi * f * t); % Quadrature component
    
    y_in = [y_in y1];
    y_qd = [y_qd y2];
    y = [y y1 + y2]; % Combine for QPSK signal
end

% Modulated signal (Tx_sig)
Tx_sig = y;
tt = T / 99 : T / 99 : (T * length(data)) / 2; % Time vector for entire signal

% Plot modulated signals
figure(2)
subplot(3, 1, 1);
plot(tt, y_in, 'linewidth', 3), grid on;
title('In-phase Component');

subplot(3, 1, 2);
plot(tt, y_qd, 'linewidth', 3), grid on;
title('Quadrature Component');

subplot(3, 1, 3);
plot(tt, Tx_sig, 'r', 'linewidth', 3), grid on;
title('QPSK Modulated Signal');

% Demodulation

Rx_data = []; % Initialize received data
Rx_sig = Tx_sig; % Assume no noise, received signal = transmitted signal

% Demodulate for each data pair
for i = 1 : length(data) / 2
    % In-phase detection
    Z_in = Rx_sig((i - 1) * length(t) + 1 : i * length(t)) .* cos(2 * pi * f * t);
    Z_in_intg = (trapz(t, Z_in)) * (2 / T); % Integrate and make decision
    Rx_in_data = Z_in_intg > 0;

    % Quadrature detection
    Z_qd = Rx_sig((i - 1) * length(t) + 1 : i * lehngth(t)) .* sin(2 * pi * f * t);
    Z_qd_intg = (trapz(t, Z_qd)) * (2 / T); % Integrate and make decision
    Rx_qd_data = Z_qd_intg > 0;

    % Store received bits
    Rx_data = [Rx_data Rx_in_data Rx_qd_data];
end

% Plot received data
figure(3)
stem(Rx_data, 'linewidth', 3);
title('Information after Receiving');
axis([0 11 0 1.5]), grid on;
`
            },
            {
                header: "Prg 7",
                question: "DC Program Question Placeholder",
                code: 
`M = 16;               % Modulation order
k = log2(M);           % Number of bits per symbol
n = 30000;             % Number of symbols per frame
sps = 1;               % Number of samples per symbol

% Use default random number generator
rng default

% Generate vector of binary data
dataIn = randi([0 1],n*k,1);

% Plot random bits
stem(dataIn(1:40), 'filled');
title('Random Bits');
xlabel('Bit Index');
ylabel('Binary Value');

% Convert binary data to integer symbols
dataSymbolsIn = bit2int(dataIn,k);

% Plot random symbols
figure;
stem(dataSymbolsIn(1:10));
title('Random Symbols');
xlabel('Symbol Index');
ylabel('Integer Value');

% Binary-encoded QAM modulation
dataMod = qammod(dataSymbolsIn, M,'bin');

% Gray-encoded QAM modulation
dataModG = qammod(dataSymbolsIn, M);

% Set Eb/No
EbNo = 10;

% Convert Eb/No to SNR
snr = convertSNR(EbNo, 'ebno', samplespersymbol=sps, bitspersymbol=k);

% Add AWGN noise to the modulated signals
receivedSignal = awgn(dataMod, snr, 'measured');
receivedSignalG = awgn(dataModG, snr, 'measured');

% Plot the constellation diagrams
sPlotFig = scatterplot(receivedSignal,1,0,'g.');
hold on
scatterplot(dataMod,1,0,'k*',sPlotFig)

% Binary-encoded QAM demodulation
dataSymbolsOut = qamdemod(receivedSignal,M,'bin');

% Gray-coded QAM demodulation
dataSymbolsOutG = qamdemod(receivedSignalG,M);

% Convert integer symbols back to binary data
dataOut = int2bit(dataSymbolsOut,k);
dataOutG = int2bit(dataSymbolsOutG,k);

% Calculate bit error rate for binary-coded QAM
[numErrors,ber] = biterr(dataIn,dataOut);
fprintf('\nThe binary coding bit error rate is %5.2e, based on %d errors.\n', ber,numErrors)

% Calculate bit error rate for Gray-coded QAM
[numErrorsG,berG] = biterr(dataIn, dataOutG);
fprintf('\nThe Gray coding bit error rate is %5.2e, based on %d errors.\n', berG,numErrorsG)

% Modulation order
M = 16;

% Integer input
x = (0:15);

% 16-QAM output (binary-coded)
symbin = qammod(x,M,'bin');

% 16-QAM output (Gray-coded)
symgray = qammod(x,M,'gray');

% Plot constellation diagram for Gray-coded QAM
scatterplot(symgray,1,0,'b*');

% Add labels to the constellation points
for k = 1:M
    text(real(symgray(k))-0.0,imag(symgray(k))+0.3,dec2base(x(k),2,4), 'Color',[0 1 0]);
    text(real(symgray(k))-0.5,imag(symgray(k))+0.3,num2str(x(k)), 'Color',[0 1 0]);
    text(real(symbin(k))-0.0,imag(symbin(k))-0.3,dec2base(x(k),2,4), 'Color',[1 0 0]);
    text(real(symbin(k))-0.5,imag(symbin(k))-0.3,num2str(x(k)), 'Color',[1 0 0]);
end

% Set plot title and axis limits
title('16-QAM Symbol Mapping')
axis([-4 4 -4 4])`
            },        
            // Additional DC programs can be added here
            /*
            {
                header: "Additional DC Program Placeholder",
                question: "Additional DC Program Question Placeholder",
                code: 
`% Additional DC Program Code Placeholder`
            },
            */
        ],
        
        // Future placeholders for another subject set (commented out)
        /*
        AnotherSubject: [
            {
                header: "Another Subject Program Placeholder",
                question: "Another Subject Program Question Placeholder",
                code: 
`% Another Subject Program Code Placeholder`
            },
            // Additional programs for AnotherSubject can be added here
            {
                header: "Additional AnotherSubject Program Placeholder",
                question: "Additional AnotherSubject Program Question Placeholder",
                code: 
`% Additional AnotherSubject Program Code Placeholder`
            }
        ]
        */
    }
}
