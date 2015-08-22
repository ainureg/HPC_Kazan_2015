#include<stdlib.h>
#include<stdio.h>
#include <math.h>

double time_tot = 0.1;        // total time 
double dt =  0.0002;          // time step
double h = 0.02;              // step on the x-coordinate
double l = 1;                 // length of the rod
double k = 1;                 // koeff of the thermal diffusivity
double u_0 = 1;               // initial value
double pi = 3.14159265358;
 

int main(int argc, char *argv[]){
	int i, m, nPoints;
	double time, x, a, sum;
	double *u_prev, *u_next, *u_exact;
	
	
	nPoints =  l / h - 1;
	
	u_next = (double*) malloc(sizeof(double) * (nPoints+2));
	u_prev = (double*) malloc(sizeof(double) * (nPoints+2));
	u_exact = (double*) malloc(sizeof(double) * (nPoints+2));

	printf ("Number of inner points N = %d \n", nPoints);
	printf("Number of time steps = %f \n", time_tot / dt);

       //--------- initial conditions ----------------------------
	for (i = 1; i <= nPoints; i++){
		u_prev[i] = u_0;
	}	
       time = 0;

	// -------------- time circle ------------------------------
       while (time < time_tot){
		for (i = 1; i <= nPoints; i++){
	       // -------- finite difference scheme -------------
	              u_next[i] = u_prev[i] + k * dt / (h * h) * (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]);
		}
		for (i = 1; i <= nPoints; ++i){		
			u_prev[i] = u_next[i];
		}
		time = time + dt;
	}		
	printf("Numerical solution: \n");
	for (i = 1; i <= nPoints; i++){
		printf("%lf  ", u_prev[i]);
	}
	printf("\n");

	// --------------- end of time circle -----------------------
	
	// ------------- exact solution -----------------------------
	
	printf("Exact solution: \n");
	for (i = 0; i < nPoints; i++){
		x = (i + 1) * h;
		sum = 0;
		for (m = 0; m < 3; m++){
      			a =  exp(- pi * pi * (2*m+1) * (2*m+1) * time_tot) * sin( pi * (2*m+1) * x / l) / (2*m+1);
			sum = sum + 4 * u_0 * a/ pi;
		}
		u_exact[i] = sum;
		printf("%f  ", u_exact[i]);
	}
	printf("\n");

	free(u_next);
	free(u_prev);
	free(u_exact);

	return 0;
}
