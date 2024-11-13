const programData = {
    subjects: {
        DSP: [
            {
                header: "DSP Program Placeholder",
                question: "DSP Program Question Placeholder",
                code: `% DSP Program Code Placeholder`
            },
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
