#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char *argv[]){
double tol=0.000001;
int k=1000;

int i;
double S,Q;
double begin, end, new;

int myrank, size;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &size);

double h=1/(double)(k-1)/(double)size;

MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

MPI_Status Status;

begin= myrank/(double)(size);
for (i=0;i<k-1;i++){
	end=begin+h;
	new=(1/(1+begin*begin) +1/(1+end*end))*2*h;
	S=S+new;
	begin=end;
	}
MPI_Reduce(&S,&Q, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if (myrank==0){
	printf("%10.10f\n",Q);
}

MPI_Finalize();
return 0;
}
