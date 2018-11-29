clc
clear all;
load('203_#5b_meitu_10.pngI_edge.mat');
load('203_#5b_meitu_10.pngI_tiny.mat');
figure;imshow(I_tiny);
figure;imshow(I_edge);
widd = 26;
thresh=[0.01,0.17];
sigma=2;%定义高斯参数
[p,q] = size(I_edge);
f = edge(double(I_edge(widd:end-widd,widd:end-widd)),'canny',thresh,sigma);
figure;imshow(f,[]);
f_fill = imfill(f,'holes');
figure;imshow(f_fill,[]);
[L, num] = bwlabel(f_fill,8);
figure;imshow(L,[]);
mama = 0;
for i = 1:1:num
    [cunx,cuny] = find(L==i);
    rato = length(cunx)/sqrt(p*q);
    if rato>mama
        mama = rato;
    end
    if (rato<1)&&(max(max(cunx)-min(cunx),max(cuny)-min(cuny))<90)
        I_tiny(cunx+widd,cuny+widd) = I_tiny(cunx+widd,cuny+widd)+I_edge(cunx+widd,cuny+widd);
        I_edge(cunx+widd,cuny+widd) = 0;
    end
end
figure;imshow(I_edge);
figure;imshow(I_tiny);
% savePath = ['refine_203_#5b_meitu_10.pngI_edge.mat'];
% save(savePath,'I_edge');
% savePath = ['refine_203_#5b_meitu_10.pngI_tiny.mat'];
% save(savePath,'I_tiny');

