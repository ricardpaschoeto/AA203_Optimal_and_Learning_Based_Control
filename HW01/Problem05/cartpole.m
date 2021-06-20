function K = cartpole()

syms mp mc l g dt x teta x_dot teta_dot u dt;

H = [mc + mp mp*l*cos(teta);mp*l*cos(teta) mp*l^2];
C = [0 -mp*l*teta_dot*sin(teta);0 0];
G = [0 mp*g*l*sin(teta)]';
B = [1 0]';

a21 = -inv(H)*diff(G,teta);
a22 = -inv(H)*C;

b21 = inv(H)*B;

A = [0 0 1 0;0 0 0 1;0 a21(1) a22(1) 0;0 a21(2) a22(2) 0]*dt;
B = [0 0 b21']*dt;

A = double(subs(A,[mp, mc, l, g, x, teta, x_dot, teta_dot, u, dt],[2.0, 10.0, 1.0, 9.81, 0, pi, 0, 0, 0, 0.1]));
B = double(subs(B,[mp, mc, l, g, x, teta, x_dot, teta_dot, u, dt],[2.0, 10.0, 1.0, 9.81, 0, pi, 0, 0, 0, 0.1]))';

[~,K,~] = idare(A, B, eye(4), eye(1));

end