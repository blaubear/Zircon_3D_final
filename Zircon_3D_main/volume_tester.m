0.042625444874541 * 1 

qwe = (S_right(1:n)-S_left(1:n)).*(S_top(1:n)-S_bot(1:n)) - S*S;

new = (sum(qwe) / (0.15*0.15) )* 1.e+6

%L_size/nx*L_size/nx *n* 1.e+6