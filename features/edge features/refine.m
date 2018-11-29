clc
clear all;
load('203_#5b_meitu_10.pngI_edge.mat');
load('203_#5b_meitu_10.pngI_rond.mat');
load('203_#5b_meitu_10.pngI_tiny.mat');
widd = 16;
thresh=[0.01,0.17];
sigma=2;%定义高斯参数
[p,q] = size(I_edge);
f = edge(double(I_edge(28:end-28,28:end-28)),'canny',thresh,sigma);
[H, theta, rho]= hough(f,'RhoResolution', 0.5);
peak=houghpeaks(H,30,'Threshold',1);
lines=houghlines(f,theta,rho,peak,'FillGap',5,'MinLength',20);
figure,imshow(f,[]),title('Hough Transform Detect Result'),hold on
for k=1:length(lines)
    xy=[lines(k).point1;lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color',[.6 .6 .6]);
end
loc = [];
theta = [];
for i = 1:1:length(lines)
    x = floor((lines(i).point1+lines(i).point2)/2);
    if (lines(i).theta<=22.5)&&(lines(i).theta>=-22.5)
        I_temp = uint8(zeros(p,q));
        I_temp(widd:end-widd,max(widd,x(2)-10):min(q-widd,x(2)+10)) = 1;
        figure;imshow(I_temp*255);
        cou = find(I_edge(widd:end-widd,max(widd,x(2)-10):min(q-widd,x(2)+10))>50);
        [m,n] = size(I_edge(widd:end-widd,max(widd,x(2)-10):min(q-widd,x(2)+10)))
        rati = length(cou)/m/n;
        if rati>(1/4)
        else
            I_tiny(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)))...
                = I_tiny(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)))...
                +I_edge(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)));
            I_edge(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1))) = 0;
        end
    end
    if (lines(i).theta>=22.5)&&(lines(i).theta<=67.5)
        xx = x(2);
        y = x(1);
        I_temp = uint8(zeros(p,q));
        while((xx<=p-widd)&&(y<=q-widd)&&(xx>=widd)&&(y>=widd))
            I_temp(max(widd,xx-10):min(xx+10,q-widd),y) = 1;
            xx = xx+1;
            y = y-1;
        end
        xx = x(2);
        y = x(1);
        while((xx<=p-widd)&&(y<=q-widd)&&(xx>=widd)&&(y>=widd))
            I_temp(max(widd,xx-10):min(xx+10,q-widd),y) = 1;
            xx = xx-1;
            y = y+1;
        end
        figure;imshow(I_temp*255);
        cou = find(I_temp.*I_edge>50);
        sa = find(I_temp>0);
        rati = length(cou)/sa;
        if rati>(1/4)
        else
            I_tiny(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)))...
                = I_tiny(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)))...
                +I_edge(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)));
            I_edge(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1))) = 0;
        end
    end
    if (lines(i).theta>=67.5)&&(lines(i).theta<=-67.5)
        I_temp = uint8(zeros(p,q));
        I_temp(max(widd,x(1)-10):min(p-widd,x(1)+10),widd:end-widd) = 1;
        figure;imshow(I_temp*255);
        cou = find(I_edge(max(widd,x(1)-10):min(p-widd,x(1)+10),widd:end-widd)>50);
        [m,n] = size(I_edge(max(widd,x(1)-10):min(p-widd,x(1)+10),widd:end-widd));
        rati = length(cou)/m/n;
        if rati>(1/4)
        else
            I_tiny(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)))...
                = I_tiny(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)))...
                +I_edge(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)));
            I_edge(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1))) = 0;
        end
    end
    if (lines(i).theta<=-22.5)&&(lines(i).theta>=-67.5)
        xx = x(2);
        y = x(1);
        I_temp = uint8(zeros(p,q));
        while((xx<=p-widd)&&(y<=q-widd)&&(xx>=widd)&&(y>=widd))
            I_temp(max(widd,xx-10):min(xx+10,q-widd),y) = 1;
            xx = xx+1;
            y = y+1;
        end
        xx = x(1);
        y = x(2);
        while((xx<=p-widd)&&(y<=q-widd)&&(xx>=widd)&&(y>=widd))
            I_temp(max(widd,xx-10):min(xx+10,q-widd),y) = 1;
            xx = xx-1;
            y = y-1;
        end
        figure
        imshow(I_temp*255);
        cou = find(I_temp.*I_edge>50);
        sa = find(I_temp>0);
        rati = length(cou)/sa;
        if rati>(1/4)
        else
            I_tiny(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)))...
                = I_tiny(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)))...
                +I_edge(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1)));
            I_edge(min(lines(i).point1(2),lines(i).point2(2)):max(lines(i).point1(2),lines(i).point2(2)),min(lines(i).point1(1),lines(i).point2(1)):max(lines(i).point1(1),lines(i).point2(1))) = 0;
        end
    end
end
figure;imshow(I_edge);
figure;imshow(I_tiny);
figure;imshow(I_rond);
