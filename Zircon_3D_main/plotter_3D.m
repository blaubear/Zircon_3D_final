function plotter_3D(xxx_slide, yyy_slide, zzz_slide,X,Y,Z,AA,n,S_right,S_left,S_top,S_bot,S_forth,S_back,L_size,FLAG_numbers)
        xslice = [ xxx_slide ];
        yslice = [ yyy_slide];
        zslice = [0, zzz_slide]; 
        slice(X,Y,Z,AA,xslice,yslice,zslice);
        xlabel('x')
ylabel('y')
zlabel('z')
        axis image;
        c = colorbar;
        
        drawnow;
        c.Limits = [160 200];
        colormap(jet);
        %light;
        %shading interp;
        alpha 0.7
if(FLAG_numbers ==1)
      str_N_cryst           = {'N cryst=',num2str(n)};
 
      text(0,-0.85,str_N_cryst);


      for j=1:n
          text((S_right(j) + S_left(j))/2/L_size,(S_top(j) + S_bot(j))/2/L_size,(S_forth(j) + S_back(j))/2/L_size,num2str(j), 'Color', 'black','FontSize', 10);
      end
end
end