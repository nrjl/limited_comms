nice_colors_lines = [57,106,177; 218,124,48;62,150,81;204,37,41;83,81,84;107,76,154;146,36,40;148,139,61]./255;
nice_colors_bars = [114,147,203;225,151,76;132,186,91;211,94,96;128,133,133;144,103,167;171,104,87;204,194,16]./255;

folders = {'2016_10_27-11_04', '2016_10_26-17_11', '2016_10_31-13_39'};

dir_prefix = '../data/';
findices = [[1,4,3];[1,4,3];[1,3,2]];

n_methods = size(findices, 2);
intergroup_spacing = 0.3;
intermethod_spacing = 0.1;
clear_space = intermethod_spacing/(n_methods-1);
widths = (1-intergroup_spacing-clear_space)/n_methods;

positions = 1:numel(folders);

figure(2); clf;
set(gcf, 'NextPlot', 'add');
for i = positions
    data = readtable(strcat(dir_prefix, folders{i}, '/obsterm.csv'));
    for j = 1:size(findices, 2)
        dd = data(:,findices(i,j));
        boxplot(dd{:,:}, 'color', nice_colors_lines(j,:), 'width', widths, 'positions', (i-1)+intergroup_spacing/2+widths/2+(j-1) *(widths+clear_space));
        set(gca, 'nextplot', 'add');
    end
end
set(gca, 'XTick', (1:n_methods)-0.5);
xlabel('Number of states (|\Theta|)', 'Interpreter','Tex');
ylabel('Observations to find target (\epsilon = 0.02)', 'Interpreter','Tex');
set(gca, 'XTickLabel', {'64','128','256'});
set(gca, 'xlim', [0,n_methods])
set(gca, 'ylim', [0,200])