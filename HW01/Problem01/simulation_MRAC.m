function [] = simulation_MRAC(y0, ym0, kr0, ky0, value)
alpha = -1;
beta = 3;
alpha_m = 4;
beta_m = 4;
dt = 0.01;
gamma = 2.0;
tval = linspace(0,10,(10/dt));
y = y0;
ym = ym0;
kr = kr0;
ky = ky0;

y_history = zeros(1,size(tval,2));
ym_history = zeros(1,size(tval,2));
kr_history = zeros(1,size(tval,2));
ky_history = zeros(1,size(tval,2));
error_history = zeros(1,size(tval,2));
delta_r_hist = zeros(1,size(tval,2));
delta_y_hist = zeros(1,size(tval,2));

y_history(1) = y;
ym_history(1) = ym;
kr_history(1) = kr;
ky_history(1) = ky;

kr_star = beta_m/beta;
ky_star = (alpha - alpha_m)/beta;
kr_star_hist = kr_star*ones(1,size(tval,2));
ky_star_hist = ky_star*ones(1,size(tval,2));

error = y0 - ym0;
error_history(1) = error;
dr = kr0 - kr_star;
delta_r_hist(1) = dr;
dy = ky0 - ky_star;
delta_y_hist(1) = dy;
index = 2;

for t = tval(:,2:size(tval,2))    
    r = ref(t, value);
    y = y + y_dot(y, r, ky, kr)*dt;
    y_history(index) = y;
    ym = ym + ym_dot(ym, r)*dt;
    ym_history(index) = ym;
    error = y - ym;
    error_history(index) = error;
    kr = kr + kr_dot(gamma, error, r)*dt;
    kr_history(index) = kr;
    ky = ky + ky_dot(gamma, error, y)*dt;
    ky_history(index) = ky;
    
    delta_r = kr - kr_star;
    delta_r_hist(index) = delta_r;
    delta_y = ky - ky_star;
    delta_y_hist(index) = delta_y;
    
    index = index + 1;
end
plot_outputs(tval, y_history, ym_history);
plot_gains(tval,kr_history, ky_history, kr_star_hist, ky_star_hist);
end

function [] = plot_outputs(t, y_history, ym_history)
    figure;
    hold on;
    plot(t, y_history,'b', 'LineWidth',1);
    plot(t, ym_history,'r--', 'LineWidth',2);
    title('True Model and Reference Model');
    xlabel('time');
    ylabel('Outputs');
    legend('y(t) - True Model','ym(t) - Reference Model');
    grid on;
end

function [] = plot_gains(t, kr_history, ky_history, kr_star_hist, ky_star_hist)
    figure;
    hold on;
    plot(t, kr_history,'b', 'LineWidth',1);
    plot(t, kr_star_hist,'b--', 'LineWidth',2);
    plot(t, ky_history,'r', 'LineWidth',1);
    plot(t, ky_star_hist,'r--', 'LineWidth',2);
    title('Gains Time history');
    xlabel('time');
    ylabel('Gains')
    legend('kr', 'kr*', 'ky','ky*');
    grid on;
end

function r = ref(t, value)
   if value == 0
      r = 4;
   elseif value == 1
      r = 4*sin(3*t); 
   end
end

function yd = y_dot(y, r, ky, kr)
    yd = 3*control(y, r, ky, kr) + y; 
end

function u = control(y, r, ky, kr)
    u = kr*r + ky*y; 
end

function ymd = ym_dot(ym, r)
    ymd = 4*(r - ym);
end

function krd = kr_dot(gamma, error, r)
    krd = -gamma*error*r;
end

function kyd = ky_dot(gamma, error, y)
    kyd = -gamma*error*y;
end