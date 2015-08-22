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

if(myrank==0){
	scanf("%d",&n);
	n++;
	}
MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

array = (int*)malloc((n) * sizeof(int));

int *sendcounts,*displs;

sendcounts = (int*) malloc(sizeof(int) * (size));
displs= (int*) malloc(sizeof(int) * (size));
	
for (i=0;i<size;i++){
	sendcounts[i]=(int)n/size+((int) (i< (n % size)));
		}

displs[0]=0;

for (i=1;i<size;i++){
	displs[i]=displs[i-1]+sendcounts[i-1];
		}

if (myrank==0){	
	for (i=0;i<n;i++){
		array[i]=i;
		}
	}

a=(int*)malloc((sendcounts[myrank])*sizeof(int));
MPI_Scatterv(array,sendcounts, displs, MPI_INT,a, n, MPI_INT, 0,MPI_COMM_WORLD);

for(i=0;i<sendcounts[myrank];i++){
	sum+=a[i];
	}
	
MPI_Reduce(&sum,&S, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

if (myrank==0){
	printf("Sum is %d\n",S);
	}
MPI_Finalize();
return 0;
}
