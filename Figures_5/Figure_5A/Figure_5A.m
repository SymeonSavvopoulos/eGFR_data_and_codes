clc
clear

%All patients in set 1 (N1)
[data_min_thr_mean_set_1,data_mean_thr_mean_set_1,data_max_thr_mean_set_1,data_min_thr_std_set_1,data_mean_thr_std_set_1,data_max_thr_std_set_1, f1_set_1]=studies_set_1()

%All patients in set 2 (N2)
[data_min_thr_mean_set_2,data_mean_thr_mean_set_2,data_max_thr_mean_set_2,data_min_thr_std_set_2,data_mean_thr_std_set_2,data_max_thr_std_set_2, f1_set_2]=studies_set_2()


figure(1)
xlim([0 5])

x1=[2:1:4];
y1=[data_min_thr_mean_set_1,data_mean_thr_mean_set_1,data_max_thr_mean_set_1];
err1=[data_min_thr_std_set_1,data_mean_thr_std_set_1,data_max_thr_std_set_1];

errorbar(x1,y1,err1,"o")
names = {'GFR_{min}'; 'GFR_{mean}'; 'GFR_{max}'};
set(gca,'xtick',[2:4],'xticklabel',names)

hold on

x2=[2.2:1:4.2];
y2=[data_min_thr_mean_set_2,data_mean_thr_mean_set_2,data_max_thr_mean_set_2];
err2=[data_min_thr_std_set_2,data_mean_thr_std_set_2,data_max_thr_std_set_2];
errorbar(x2,y2,err2,"o")
names = {'GFR_{min}'; 'GFR_{mean}'; 'GFR_{max}'};
set(gca,'xtick',[2.2:4.2],'xticklabel',names)



for i=1:9
min_f1_1(i)=mean(f1_set_1(i,2:end));
std_f1_1(i)=std(f1_set_1(i,2:end));
end

minf1=mean([min_f1_1(1),min_f1_1(2) min_f1_1(3)]);
minstdf1=mean([std_f1_1(1),std_f1_1(2) std_f1_1(3)]);

meanf1=mean([min_f1_1(4),min_f1_1(5) min_f1_1(6)]);
meanstdf1=mean([std_f1_1(4),std_f1_1(5) std_f1_1(6)]);

maxf1=mean([min_f1_1(7),min_f1_1(8) min_f1_1(9)]);
maxstdf1=mean([std_f1_1(7),std_f1_1(8) std_f1_1(9)]);


xf1=[minf1,meanf1,maxf1]';
yf1=[minstdf1,meanstdf1,maxstdf1]';



for i=1:9
min_f1_2(i)=mean(f1_set_2(i,2:end));
std_f1_2(i)=std(f1_set_2(i,2:end));
end

minf2=mean([min_f1_2(1),min_f1_2(2) min_f1_2(3)]);
minstdf2=mean([std_f1_2(1),std_f1_2(2) std_f1_2(3)]);

meanf2=mean([min_f1_2(4),min_f1_2(5) min_f1_2(6)]);
meanstdf2=mean([std_f1_2(4),std_f1_2(5) std_f1_2(6)]);

maxf2=mean([min_f1_2(7),min_f1_2(8) min_f1_2(9)]);
maxstdf2=mean([std_f1_2(7),std_f1_2(8) std_f1_2(9)]);


xf2=[minf2,meanf2,maxf2]';
yf2=[minstdf2,meanstdf2,maxstdf2]';


figure(2)
xlim([0 5])

x1=[2:1:4];
y1=[minf1,meanf1,maxf1];
err1=[minstdf1,meanstdf1,maxstdf1];

errorbar(x1,y1,err1,"o")
names = {'GFR_{min}'; 'GFR_{mean}'; 'GFR_{max}'};
set(gca,'xtick',[2:4],'xticklabel',names)

hold on

x2=[2.2:1:4.2];
y2=[minf2,meanf2,maxf2];
err2=[minstdf2,meanstdf2,maxstdf2];
errorbar(x2,y2,err2,"o")
names = {'GFR_{min}'; 'GFR_{mean}'; 'GFR_{max}'};
set(gca,'xtick',[2.2:4.2],'xticklabel',names)


