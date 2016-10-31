nice_colors_lines = [57,106,177; 218,124,48;62,150,81;204,37,41;83,81,84;107,76,154;146,36,40;148,139,61]./255;
nice_colors_bars = [114,147,203;225,151,76;132,186,91;211,94,96;128,133,133;144,103,167;171,104,87;204,194,16]./255;

good_means =[57.41, 48.92, 46.27];
good_stds = [28.693237879333175, 25.323775389937417, 24.024926638805784];
good_shares = [0.0, 2.18, 3.69];
bad_means = [135.8, 110.54, 102.79];
bad_stds = [42.782706786738025, 47.745454233884928, 46.182961143694541];
bad_shares = [0.0, 2.13, 4.45];

figure(1); clf;
hax = gca;
hbars = bar(hax, [good_means',bad_means']);
hbars(1).FaceColor = nice_colors_bars(1,:);
hbars(2).FaceColor = nice_colors_bars(2,:);
hbars(1).EdgeColor = 'none';
hbars(2).EdgeColor = 'none';
ylims = get(hax, 'Ylim');
y = (ylims(1) + diff(ylims)*0.05)*ones(1,3);
hold on;
x =  hbars(1).XData + hbars(1).XOffset;
hErrorbar(1) = errorbar(x, good_means, good_stds, good_stds, '.k');
text(x, y, cellstr(num2str(good_shares'))', 'HorizontalAlignment', 'center');

x =  hbars(2).XData + hbars(2).XOffset;
hErrorbar(2) = errorbar(x, bad_means, bad_stds, bad_stds, '.k');
text(x, y, cellstr(num2str(bad_shares'))', 'HorizontalAlignment', 'center');

ylabel(hax, 'Number of observations to find target');
set(hax, 'xticklabels', {'No sharing', '20% Max. KLD', '10% Max. KLD'});
legend(hax, {'High-quality sensor', 'Low-quality sensor'});