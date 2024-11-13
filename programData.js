const programData = {
    subjects: {
        DSP: [
            {
                header: "Gram-Schmidt Orthogonalization and Plotting",
                question: "Perform Gram-Schmidt Orthogonalization on a set of vectors and plot the result.",
                code: `clc; close all; clear all;
% Define the set of input vectors 3x3 matrix where each column is a vector
V = [1 1 0; 1 0 1; 0 1 1]'; % Linearly Independent vectors
%V = rand(3, 3);
%V = [1 2 3; 1 5 2; 2 4 6]'; %Linearly Dependent
%V = [3 2 1;1 2 3;0 1 4];
% Number of vectors
n = size(V, 2); % no of columns(vectors)
% Initialize the matrix for orthogonal vectors
U = zeros(size(V));
% Gram-Schmidt Process
for i = 1:n
    % Start with the original vector
    U(:, i) = V(:, i);

    % Subtract projections of previous orthogonal vectors
    for j = 1:i-1
        U(:, i) = U(:, i) - (dot(U(:, j), V(:, i)) / dot(U(:, j), U(:, j))) * U(:, j);
    end
end
% Normalize the orthogonal vectors to make them orthonormal
E = zeros(size(U));
for i = 1:n
    E(:, i) = U(:, i) / norm(U(:, i));
end
% Display the results
disp('Original Vectors (V):');
disp(V);
disp('Orthogonal Vectors (U):');
disp(U);
disp('Orthonormal Vectors (E):');
disp(E);
% Plotting the original vectors
figure;
hold on;
grid on;
quiver3(0, 0, 0, V(1, 1), V(2, 1), V(3, 1), 'r', 'LineWidth', 2);
quiver3(0, 0, 0, V(1, 2), V(2, 2), V(3, 2), 'g', 'LineWidth', 2);
quiver3(0, 0, 0, V(1, 3), V(2, 3), V(3, 3), 'b', 'LineWidth', 2);
% Plotting the orthonormal vectors
quiver3(0, 0, 0, E(1, 1), E(2, 1), E(3, 1), 'r--', 'LineWidth', 2);
quiver3(0, 0, 0, E(1, 2), E(2, 2), E(3, 2), 'g--', 'LineWidth', 2);
quiver3(0, 0, 0, E(1, 3), E(2, 3), E(3, 3), 'b--', 'LineWidth', 2);
% Setting up the plot
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Gram-Schmidt Orthonormalization');
legend({'V1', 'V2', 'V3', 'E1', 'E2', 'E3'}, 'Location', 'Best');
axis equal;
hold off`
            },
            // Additional DSP programs can be added here
        ],
        DC: [
            {
                header: "Example DC Program",
                question: "This is an example question for Digital Communication.",
                code: `% Example code for Digital Communication
% Here you would add a sample MATLAB code related to Digital Communication`
            }
            // Additional DC programs can be added here
        ]
    }
};
