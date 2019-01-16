#LOSS AND GRADLOSS (bilog)
set xlabel "t"
set ylabel "Loss, D_1, D_2"
set logs
set tics format "10^{%T}"
set key bottom left
#More points
p "< awk '(NR>1){print $3+$4,$0}' cifar10_gradloss.txt" u 1:9 w lp t "Train Loss", "" u 1:11 w lp t "D_1", "" u 1:12 w lp t "D_2"
rep "cifar10_loss.txt" u 1:4 w lp t "Train Loss(tbar)"
#Less points
p "< awk '(NR>1 && $2==0){print $3+$4,$0}' cifar10_gradloss.txt" u 1:9 w lp t "Train Loss", "" u 1:11 w lp t "D_1", "" u 1:12 w lp t "D_2"
rep "cifar10_loss.txt" u 1:4 w lp t "Train Loss(tbar)"



#LOSS AND GRADLOSS (semilog)
set xlabel "t"
set ylabel "Loss, D_1, D_2"
unset logs; set logs x
set xtics format "10^{%T}"
set ytics format "%g"
set key top left
p [][:2.5]"< awk '(NR>1){print $3+$4,$0}' cifar10_gradloss.txt" u 1:9 w lp t "Train Loss", "" u 1:11 w lp t "D_1", "" u 1:12 w lp t "D_2"



#CORRELATION FUNCTIONS
model="conv1020relu"; dataset="cifar10"; bs=100; lr=0.01; m=0
set title sprintf("%s - %s - B=%g, LR=%g, m=%d",model,dataset,bs,lr,m)
set xlabel "t"
set ylabel "C(t_w,t_w+t)"
set tics format "10^{%T}"
set key bottom right
set logs
unset colorbox
p [:] for [i=0:49]  sprintf("<awk '($1==%d)' cifar10_gradloss.txt|sort -nk2",i) u ($4):($5) w p t column(3)  linecolor palette frac (i)/(49.0)

#CORRELATION FUNCTIONS rescaled with tw
set xlabel "t/t_w"
set ylabel "C(t_w,t_w+t)"
set tics format "10^{%T}"
set logs
unset colorbox
p for [i=0:49]  sprintf("<awk '($1==%d)' cifar10_gradloss.txt|sort -nk2",i) u ($4/$3):($5) w p t column(3) linecolor palette frac (i)/(49.0)


#CORRELATION FUNCTIONS rescaled with D2
set xlabel "t"
set ylabel "C(t_w,t_w+t)/D_2(t_w)"
set tics format "10^{%T}"
set logs
unset colorbox
p [:1e7] for [i=0:49]  sprintf("<awk '($1==%d)' cifar10_gradloss.txt|sort -nk2",i) u ($4):($5/$11) w p t column(3) linecolor palette frac (i)/(49.0)


#PLOT D2(tw)
set xlabel "t"
set ylabel "D_2(t_w)"
set tics format "10^{%T}"
set logs
p "<awk '$2==0{print $0}' cifar10_gradloss.txt" u 1:11 w lp



#PLOT D2(tw,tw+t)
set xlabel "t"
set ylabel "D_2(t_w,t_w+t)"
set tics format "10^{%T}"
set logs
unset colorbox
p [:1e7] for [i=0:49]  sprintf("<awk '($1==%d)' cifar10_gradloss.txt|sort -nk2",i) u ($4):($11) w p t column(3) linecolor palette frac (i)/(49.0)



#A SLOPPY INTEGRAL OF THE LOSS
set xlabel "t"
set ylabel "Loss"
set logs
p\
"<awk 'BEGIN{t_old=0; A=0;}(NR>1){delta=$1-t_old; dA=delta*$4; A+=dA; print $1,delta,$4,A; t_old=$1}' cifar10_loss.txt" u 1:3 t "Loss",\
"<awk 'BEGIN{t_old=0; A=0;}(NR>1){delta=$1-t_old; dA=delta*$4; A+=dA; print $1,delta,$4,A; t_old=$1}' cifar10_loss.txt" u 1:4 t "integral(Loss)"


#PLOT integral_tw^{tw+t} D2(t')dt'
set xlabel "t"
set ylabel "D_2(t_w,t_w+t)"
set tics format "10^{%T}"
set logs
unset colorbox
p [:1e7] for [i=0:49]  sprintf("<awk 'BEGIN{t_old=0; A=0;}($1==%d){delta=$4-t_old; dA=delta*$11; A+=dA; print %d,$4,delta,$11,A; t_old=$4}' cifar10_gradloss.txt",i,i) u ($2):($4) with points notitle linecolor palette frac (i)/(49.0)
rep for [i=0:49]  sprintf("<awk 'BEGIN{t_old=0; A=0;}($1==%d){delta=$4-t_old; dA=delta*$11; A+=dA; print %d,$4,delta,$11,A; t_old=$4}' cifar10_gradloss.txt",i,i) u ($2):($5) with points notitle linecolor palette frac (i)/(49.0)
rep for [i=0:49]  sprintf("<awk '($1==%d)' cifar10_gradloss.txt|sort -nk2",i) u ($4):($5) w p t column(3)  linecolor palette frac (i)/(49.0)






#A SLOPPY INTEGRAL OF D2
sort -nk1 cifar10_Cgradloss.txt | awk 'BEGIN{tw_old=0; A=0;}(NR>1 && $2==0){deltat=$3-tw_old; dA=deltat*$9; A+=dA; print $3,deltat,$9,A; tw_old=$3}'

set xlabel "t_w"
set ylabel "Observable"
set logs
p\
"<sort -nk1 cifar10_Cgradloss.txt | awk 'BEGIN{tw_old=0; A=0;}(NR>1 && $2==0){deltat=$3-tw_old; dA=deltat*$9; A+=dA; print $3,deltat,$9,A; tw_old=$3}'" u 1:3 t "D_2(t_w)",\
"<sort -nk1 cifar10_Cgradloss.txt | awk 'BEGIN{tw_old=0; A=0;}(NR>1 && $2==0){deltat=$3-tw_old; dA=deltat*$9; A+=dA; print $3,deltat,$9,A; tw_old=$3}'" u 1:4 t "int@_0^{t_w} D_2(t) dt",\
"<sort -nk1 cifar10_Cgradloss.txt | awk 'BEGIN{tw_old=0; A=0;}(NR>1 && $2==0){deltat=$3-tw_old; dA=deltat*$9; A+=dA; print $3,deltat,$9,A; tw_old=$3}'" u 1:($3*$1) t "D_2(t_w) t_w"



sort -nk1 cifar10_Cgradloss.txt | awk 'BEGIN{tw_old=0; A=0;}(NR>1){if($2==0){ deltat=$3-tw_old; dA=deltat*$9; A+=dA; tw_old=$3} print $3,deltat,$9,A;}'





# the function integral_f(x) approximates the integral of f(x) from 0 to x.
# integral2_f(x,y) approximates the integral from x to y.
# define f(x) to be any single variable function
#
# the integral is calculated using Simpson's rule as 
#          ( f(x-delta) + 4*f(x-delta/2) + f(x) )*delta/6
# repeated x/delta times (from x down to 0)
#
delta = 0.025
#  delta can be set to 0.025 for non-MSDOS machines
#
# integral_f(x) takes one variable, the upper limit.  0 is the lower limit.
# calculate the integral of function f(t) from 0 to x
# choose a step size no larger than delta such that an integral number of
# steps will cover the range of integration.
integral_f(x) = (x>0)?int1a(x,x/ceil(x/delta)):-int1b(x,-x/ceil(-x/delta))
int1a(x,d) = (x<=d*.1) ? 0 : (int1a(x-d,d)+(f(x-d)+4*f(x-d*.5)+f(x))*d/6.)
int1b(x,d) = (x>=-d*.1) ? 0 : (int1b(x+d,d)+(f(x+d)+4*f(x+d*.5)+f(x))*d/6.)
#
# integral2_f(x,y) takes two variables; x is the lower limit, and y the upper.
# calculate the integral of function f(t) from x to y
integral2_f(x,y) = (x<y)?int2(x,y,(y-x)/ceil((y-x)/delta)): \
                        -int2(y,x,(x-y)/ceil((x-y)/delta))
int2(x,y,d) = (x>y-d*.5) ? 0 : (int2(x+d,y,d) + (f(x)+4*f(x+d*.5)+f(x+d))*d/6.)

set autoscale
set title "approximate the integral of functions"
set samples 50
set key bottom right

f(x) = exp(-x**2)

plot [-5:5] f(x) title "f(x)=exp(-x**2)", \
  2/sqrt(pi)*integral_f(x) title "erf(x)=2/sqrt(pi)*integral_f(x)", \
  erf(x) with points

