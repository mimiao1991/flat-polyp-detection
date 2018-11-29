clc
clear all;
% load('203_#5b_meitu_10.pngI_rond.mat');
% load('refine_203_#5b_meitu_10.pngI_tiny.mat');
I_tiny = imread('#1_tiny.png');
I_rond = imread('#1_rond.png');
I_tiny = rgb2gray(I_tiny);
I_rond = rgb2gray(I_rond);
%imshow(I_tiny);
%figure;imshow(I_rond);
[p,q] = size(I_tiny);
tiny_n = uint8(zeros(size(I_tiny)));
rond_n = uint8(zeros(size(I_rond)));
[nx,ny] = find(I_tiny>0);
mnx = mean(nx);
mny = mean(ny);
[rx,ry] = find(I_rond>0);
mrx = mean(rx);
mry = mean(ry);
ss = 100;
muti = 3;
for i = 1:1:length(nx)
    r_tiny = sum(sum(I_tiny(max(1,nx(i)-ss):min(p,nx(i)+ss),max(1,ny(i)-ss):min(p,ny(i)+ss))));
    r_rond = sum(sum(I_rond(max(1,nx(i)-ss):min(p,nx(i)+ss),max(1,ny(i)-ss):min(p,ny(i)+ss))));
    if (muti*r_tiny>r_rond)%&&(sqrt((nx(i)-mrx)^2+(ny(i)-mry)^2)>250)
        tiny_n(nx(i),ny(i)) = I_tiny(nx(i),ny(i));
    else
        rond_n(nx(i),ny(i)) = I_tiny(nx(i),ny(i));
    end
end
for i = 1:1:length(rx)
    r_tiny = sum(sum(I_tiny(max(1,rx(i)-ss):min(p,rx(i)+ss),max(1,ry(i)-ss):min(p,ry(i)+ss))));
    r_rond = sum(sum(I_rond(max(1,rx(i)-ss):min(p,rx(i)+ss),max(1,ry(i)-ss):min(p,ry(i)+ss))));
    if (muti*r_tiny>r_rond)%&&(sqrt((nx(i)-mrx)^2+(ny(i)-mry)^2)>250)
        tiny_n(rx(i),ry(i)) = I_rond(rx(i),ry(i));
    else
        rond_n(rx(i),ry(i)) = I_rond(rx(i),ry(i));
    end
end
figure;imshow(I_tiny);
figure;imshow(I_rond);
figure;imshow(tiny_n);
figure;imshow(rond_n);
% savePath = ['re2fine_203_#5b_meitu_10.pngI_rond.mat'];
% save(savePath,'rond_n');
% savePath = ['re2fine_203_#5b_meitu_10.pngI_tiny.mat'];
% save(savePath,'tiny_n');