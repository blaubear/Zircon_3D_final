

C_sat_start = 168.005795;
h_nd = L_size/(nx-1)
L = L_size;
tri_D = 1
if (tri_D ==1)

mass_begin = sum(C_sat_start*h_nd*nx*h_nd*nx*h_nd*nx)
%rrr = x_nd(2:end)-x_nd(1:end-1)
mass_outside =  sum(sum(sum(AA*h_nd*h_nd*h_nd)))

qwe = ((S_right(1:n)-S_left(1:n)).*(S_top(1:n)-S_bot(1:n)).*(S_forth(1:n)-S_back(1:n)) );

qwe = sum(qwe)

qwe = qwe - S*S*S*n

qccx = ( sum(qwe)) * 490000 ;

answer = (mass_outside) / mass_begin

answer = (mass_outside + qccx) / mass_begin


else
mass_begin = sum(C_sat_start*L_size*L_size)
mass_outside =  sum(sum(A*h_nd*h_nd))

qwe = (S_right-S_left).*(S_top-S_bot);

qwe = sum(qwe)

qwe = qwe - S*S*n

qccx = qwe * 490000

answer_1 = (mass_outside) / mass_begin

answer_2 = (mass_outside + qccx) / mass_begin
end



%mass_outside=  sum(sum(A)) / (nx*nx)

