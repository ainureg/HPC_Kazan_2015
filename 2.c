#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char *argv[]){

int i;
int *a,*array;
int n, sum=0, S;
int myrank, size;

MPI_Init(&argc, &argv);

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Status Status;

int k=(int)n/size+1;

a=(int*)malloc((k)*sizeof(int));
array = (int*)malloc((n) * sizeof(int));

if (myrank==0){
	scanf("%d",&n);
	n++;
	for (i=0;i<n;i++){
		array[i]=i;
	}
}
MPI_Scatter(array, k, MPI_INT,a, k, MPI_INT, 0,MPI_COMM_WORLD);

for(i=0;i<k;i++){
	sum+=a[i];
}

MPI_Reduce(&sum,&S, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

if (myrank==0){
printf("Sum is %d\n",S);
}

MPI_Finalize();
return 0;
}
