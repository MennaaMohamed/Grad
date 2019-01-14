function p = anna_phog_demo(img)
I = img;
bin = 8;
angle = 360;
L=3;
roi = [1;224;1;224];
p = anna_phog(I,bin,angle,L,roi)