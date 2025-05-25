A = dlmread('forces.txt');
t = A(:,2);
CD = A(:,3);
CL = A(:,4);

N = 1000;
D = 1/32;
u0 = 0.05;

tp = t(end-N:end);
CDp = CD(end-N:end);
CLp = CL(end-N:end);
F = fit(tp,CLp,'sin1')
fs = F.b1 / (2*pi)
St = fs*D/u0
RMS = rms(CLp)
CLm = F.a1
CLmm = max(CLp)
CDm = mean(CDp)
