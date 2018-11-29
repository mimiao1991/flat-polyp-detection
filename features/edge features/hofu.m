clc
clear all;
filename = ['/Users/miaofan/Documents/科研的内容/Pizer/polyp/texture/Jan17/huhu.png'];
file = dir(filename);
i = 1;
wname = filename;
%wname = ['/Users/miaofan/Documents/科研的内容/Pizer/polyp/texture/Jan2/203_',file(i).name];
%-------------------til here, we obtain gabor filter results-----------

area = 'area1';
filtername = 'db10';
I = imread(wname);
imshow(I);
I=rgb2gray(I);
thresh=[0.01,0.17];    
sigma=2;%定义高斯参数    
f = edge(double(I),'canny',thresh,sigma);    
figure(1),imshow(f,[]);    
title('canny 边缘检测');    
    
[H, theta, rho]= hough(f,'RhoResolution', 0.5);
figure
imshow(H,'XData',theta,'YData',rho,...
   'InitialMagnification','fit');
title('Limited Theta Range Hough Transform of Gantrycrane Image');
xlabel('\theta')
ylabel('\rho');
axis on, axis normal;
%imshow(theta,rho,H,[],'notruesize'),axis on,axis normal    
%xlabel('\theta'),ylabel('rho');    
    
peak=houghpeaks(H,10,'Threshold',1);    
hold on    
    
lines=houghlines(f,theta,rho,peak);    
figure,imshow(f,[]),title('Hough Transform Detect Result'),hold on    
for k=1:length(lines)    
    xy=[lines(k).point1;lines(k).point2];    
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color',[.6 .6 .6]);    
end    