[filename,pathname] = uigetfile('*.csv');
data = readtable(strcat(pathname,filename));
figure(1); clf;
plot(data{:,:});
legend(data.Properties.VariableNames)
