clc; close all; clear all

x = zeros(100,1);
x(40:60) = sin(linspace(0, 2*pi, 21));
figure(1)
hold on
plot(x, 'k', 'LineWidth', 6)
% A = ones(100,100).*x;
A = ones(100,100).*x + randn(100)*0.2;  % add random noise
% y = zeros(100,1);
% y(60:80) = sin(linspace(0, 2*pi, 21));
% A(:,80) = A(:,80) + y;
% figure(2)
% plot(A)
[U, S, V] = svd(A);
A_new = zeros(100,100);
p = 1;
for i = 1:p
    A_new = A_new + U(:,i)*S(i,i)*V(:,i)';
end
% A_new = U*S*V';
% figure(1)
% plot(A_new, 'LineWidth', 2)
% figure(2)
% plot(diag(S))
x_new = A_new(:,1);
plot(x_new/max(abs(x_new)), 'r', 'LineWidth', 4)
x_new2 = mean(A,2);
plot(x_new2/max(abs(x_new2)), 'b', 'LineWidth', 2)
% figure(3)
% hold on
% plot(A(:,4))
% plot(A_new(:,4))
% [V, D] = eig(A_new);
