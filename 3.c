
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<mpi.h>
double time_tot = 0.1;        // total time 
double dt =  0.0002;          // time step
double h = 0.02;              // step on the x-coordinate
double l = 1;                 // length of the rod
double k = 1;                 // koeff of the thermal diffusivity
double u_0 = 1;               // initial value
double pi = 3.14159265358;
 

int main(int argc, char *argv[]){

	
	int myrank, size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Status status;
	MPI_Request request;
	int i, m, nPoints,n;
	double time, x, a, sum;
	double *u_prev, *u_next, *u_exact,*u;
	double begin=0,end=0;

	double start, stop, total;

	start = MPI_Wtime();


	
	nPoints =  l / h-1 ;
	
	u_exact = (double*) malloc(sizeof(double) * (nPoints));
	u =       (double*) malloc(sizeof(double) * (nPoints));


	int *sendcounts,*displs;

	sendcounts = (int*) malloc(sizeof(int) * (size));
	displs= (int*) malloc(sizeof(int) * (size));
	for (i=0;i<size;i++){
		sendcounts[i]=(int)nPoints/size+((int) (i< (nPoints % size)));
			}
	displs[0]=0;
	for (i=1;i<size;i++){
		displs[i]=displs[i-1]+sendcounts[i-1];
			}


	n=sendcounts[myrank];
	
	u_next =  (double*) malloc(sizeof(double) * (n));
	u_prev =  (double*) malloc(sizeof(double) * (n));
	
	if(myrank==0){
		printf ("Number of inner points N = %d \n", nPoints);
		printf("Number of time steps = %f \n", time_tot / dt);
	}
	
	
       //--------- initial conditions ----------------------------
	for (i = 0; i < n; i++){
		u_prev[i] = u_0;
	}

	if(myrank==0){u_prev[0]=0;}
	if(myrank==size-1){u_prev[n-1]=0;}	

	time = 0;


	
	
	// -------------- time circle ------------------------------


	

       while (time < time_tot){

		if (size>1){

			if(myrank==0){
				MPI_Send((double *)&u_prev[n-1], 1, MPI_DOUBLE,1,time, MPI_COMM_WORLD);
				MPI_Recv(&end,	               1, MPI_DOUBLE,1,time, MPI_COMM_WORLD,&status);
			}else 	if(myrank==size-1)
				{
				MPI_Send((double *)&u_prev[0],   1, MPI_DOUBLE,myrank-1,time, MPI_COMM_WORLD);
				MPI_Recv(&begin,                 1, MPI_DOUBLE,myrank-1,time, MPI_COMM_WORLD, &status);
			
			}
			else if ( (myrank!=0) & (myrank!=size-1) ){		
					MPI_Send( (double *)&u_prev[0], 1, MPI_DOUBLE, myrank-1,time, MPI_COMM_WORLD);
					MPI_Recv( &begin,               1, MPI_DOUBLE, myrank-1,time, MPI_COMM_WORLD, &status);
	
					MPI_Send( (double *)&u_prev[n-1], 1, MPI_DOUBLE, myrank+1,time, MPI_COMM_WORLD);
					MPI_Recv(&end,                  1, MPI_DOUBLE, myrank+1,time, MPI_COMM_WORLD,&status);
			}
		}
		u_next[0]=u_prev[0] + k * dt / (h * h) * (u_prev[1] - 2 * u_prev[0] + begin);
		u_next[n-1]=u_prev[n-1]+ k * dt / (h * h) * (end - 2 * u_prev[n-1] + u_prev[n - 2]);
		for (i = 1; i < (n-1); i++){
	       // -------- finite difference scheme -------------
			u_next[i] = u_prev[i] + k * dt / (h * h) * (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]);
		}
		for (i = 0; i < n; ++i){		
			u_prev[i] = u_next[i];
		}
		
		time = time + dt;
	}		
		// --------------- end of time circle -----------------------


	MPI_Gatherv( u_prev, n, MPI_DOUBLE, u, sendcounts, displs, MPI_DOUBLE,0,MPI_COMM_WORLD);

	if(myrank==0){

			printf("Numerical solution: \n");
			for (i = 0; i <nPoints; i++){
					printf("%f\n  ", u[i]);
				}
				printf("\n");


		/*	printf("our 11 points: \n");
				for (i = 0; i <12; i++){
					printf("%f\n  ", u[i*(int)nPoints/12]);
				}
				printf("\n"); */

			
	
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
			stop = MPI_Wtime();
			total=stop-start;
	
			printf("total time =%f",total );
			printf("\n");
	
		}


	free(displs);
	free(sendcounts);
	free(u_next);
	free(u_prev);
	free(u_exact);
	free(u);
	
	
	MPI_Finalize();
	return 0;
}

