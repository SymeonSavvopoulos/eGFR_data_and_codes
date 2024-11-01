clc
clear

%No rejection (nr) patients in set 1 (N1)
[data_nr_mean_set_1,data_nr_std_set_1,f1_set1_nr]=studies_nr_set_1()

%One rejection (or) patients in set 1 (N1)
[data_or_mean_set_1,data_or_std_set_1,f1_set1_or]=studies_or_set_1()

%Multiple rejection (mr) patients in set 1 (N1)
[data_mr_mean_set_1,data_mr_std_set_1,f1_set1_mr]=studies_mr_set_1()


%No rejection (nr) patients in set 2 (N2)
[data_nr_mean_set_2,data_nr_std_set_2,f1_set2_nr]=studies_nr_set_2()

%One rejection (or) patients in set 2 (N2)
[data_or_mean_set_2,data_or_std_set_2,f1_set2_or]=studies_or_set_2()

%Multiple rejection (mr) patients in set 2 (N2)
[data_mr_mean_set_2,data_mr_std_set_2,f1_set2_mr]=studies_mr_set_2()



figure(1)
xlim([0 5])

x1=[1:1:3];
y1=[data_nr_mean_set_1,data_or_mean_set_1,data_mr_mean_set_1];
err1=[data_nr_std_set_1,data_or_std_set_1,data_mr_std_set_1];

errorbar(x1,y1,err1,"o")
names = {'No rejection'; 'One rejection'; 'Multiple rejections'};
set(gca,'xtick',[1:3],'xticklabel',names)

hold on

x2=[1.2:1:3.2];
y2=[data_nr_mean_set_2,data_or_mean_set_2,data_mr_mean_set_2];
err2=[data_nr_std_set_2,data_or_std_set_2,data_mr_std_set_2];
errorbar(x2,y2,err2,"o")
names = {'No rejection'; 'One rejection'; 'Multiple rejections'};
set(gca,'xtick',[1.2:3.2],'xticklabel',names)



for i=1:3
min_f1_1_nr(i)=mean(f1_set1_nr(i+6,2:end));
std_f1_1_nr(i)=std(f1_set1_nr(i+6,2:end));
end

maxf1_nr=mean([min_f1_1_nr(1),min_f1_1_nr(2) min_f1_1_nr(3)]);
maxstdf1_nr=mean([std_f1_1_nr(1),std_f1_1_nr(2) std_f1_1_nr(3)]);


for i=1:3
min_f1_1_or(i)=mean(f1_set1_or(i+6,2:end));
std_f1_1_or(i)=std(f1_set1_or(i+6,2:end));
end

maxf1_or=mean([min_f1_1_or(1),min_f1_1_or(2) min_f1_1_or(3)]);
maxstdf1_or=mean([std_f1_1_or(1),std_f1_1_or(2) std_f1_1_or(3)]);



for i=1:3
min_f1_1_mr(i)=mean(f1_set1_mr(i+6,2:end));
std_f1_1_mr(i)=std(f1_set1_mr(i+6,2:end));
end




maxf1_mr=mean([min_f1_1_mr(1),min_f1_1_mr(2) min_f1_1_mr(3)]);
maxstdf1_mr=mean([std_f1_1_mr(1),std_f1_1_mr(2) std_f1_1_mr(3)]);


xf1=[maxf1_nr,maxf1_or,maxf1_mr]';
yf1=[maxstdf1_nr,maxstdf1_or,maxstdf1_mr]';




for i=1:3
min_f1_2_nr(i)=mean(f1_set2_nr(i+6,2:end));
std_f1_2_nr(i)=std(f1_set2_nr(i+6,2:end));
end

maxf1_2_nr=mean([min_f1_2_nr(1),min_f1_2_nr(2) min_f1_2_nr(3)]);
maxstdf1_2_nr=mean([std_f1_2_nr(1),std_f1_2_nr(2) std_f1_2_nr(3)]);


for i=1:3
min_f1_2_or(i)=mean(f1_set2_or(i+6,2:end));
std_f1_2_or(i)=std(f1_set2_or(i+6,2:end));
end

maxf1_2_or=mean([min_f1_2_or(1),min_f1_2_or(2) min_f1_2_or(3)]);
maxstdf1_2_or=mean([std_f1_2_or(1),std_f1_2_or(2) std_f1_2_or(3)]);



for i=1:3
min_f1_2_mr(i)=mean(f1_set2_mr(i+6,2:end));
std_f1_2_mr(i)=std(f1_set2_mr(i+6,2:end));
end


maxf1_2_mr=mean([min_f1_2_mr(1),min_f1_2_mr(2) min_f1_2_mr(3)]);
maxstdf1_2_mr=mean([std_f1_2_mr(1),std_f1_2_mr(2) std_f1_2_mr(3)]);

xf2=[maxf1_nr,maxf1_or,maxf1_mr]';
yf2=[maxstdf1_nr,maxstdf1_or,maxstdf1_mr]';


figure(2)
xlim([0 5])

x1=[2:1:4];
y1=[maxf1_nr,maxf1_or,maxf1_mr];
err1=[maxstdf1_nr,maxstdf1_or,maxstdf1_mr];

errorbar(x1,y1,err1,"o")
names = {'No rejection'; 'One rejection'; 'Multiple rejections'};
set(gca,'xtick',[2:4],'xticklabel',names)

hold on

x2=[2.2:1:4.2];
y2=[maxf1_2_nr,maxf1_2_or,maxf1_2_mr]';
err2=[maxstdf1_2_nr,maxstdf1_2_or,maxstdf1_2_mr]';
errorbar(x2,y2,err2,"o")
names = {'No rejection'; 'One rejection'; 'Multiple rejections'};
set(gca,'xtick',[2.2:4.2],'xticklabel',names)





