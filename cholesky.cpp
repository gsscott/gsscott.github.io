#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <algorithm>

#define DEBUG_FLAG true
#define STABLE_FLAG true


void initialize_matrix_to_zero(double** A, int A_size);
void initialize_seed(double** A_seed, int A_size);
void A_from_seed(double** A, double** A_seed, int A_size);

void print_difference(double **A, double **B, int A_size);
void print_matrix(double** A, int A_size);
void stable_cholesky(double** A, double* D, double** L, int min_index, int max_index, int A_size);
void cholesky(double** A, double** L, int min_index, int max_index, int A_size);
bool check_equality(double** A, double** B, int A_size);

int main()
{
	srand(50);
	int const A_size = 50;

	//---------------------------------------------------------------------------------------------------
	//Initialize the matrices:
	//
	//A is the matrix that will be "cholesky decomposed"
	//L will be the matrix such that A = L L^T
	//
	//A_seed is a matrix used to initialize A (it's set to a lower-triangular matrix of positive values,
	//                                         then A = A_seed * (A_seed)^T gives a positive definite matrix)
	//
	//All of these matrices are symmetric (or lower triangular, for A_seed), so we only store the lower-triangular part
	//
	//Also create D, which is used only for stable_cholesky as the diagonal term
	//---------------------------------------------------------------------------------------------------
	
	std::cout << "Allocating memory for matrices. \n";
	double **A = new double*[A_size];
	double **A_seed = new double*[A_size];
	double **L = new double*[A_size];
	double *D = new double[A_size];
	for (int row = 0; row < A_size; ++row) {
		A[row] = new double[row + 1];
		A_seed[row] = new double[row + 1];
		L[row] = new double[row + 1];
		D[row] = 0;
	}

	std::cout << "Initializing matrices: Seed matrix... ";
	initialize_seed(A_seed, A_size); //set A_seed equal to a "random" lower-trianglar matrix of positive integers 1-5
	std::cout << "A matrix... ";
	A_from_seed(A, A_seed, A_size);  //set A equal to (A_seed)*(A_seed)^T
	std::cout << "L matrix... \n";
	initialize_matrix_to_zero(L, A_size);    //set L equal to the all-zero matrix
	if (DEBUG_FLAG) {
		if (A_size < 200) {
			std::cout << "The seed matrix is:" << "\n";
			print_matrix(A_seed, A_size);
			std::cout << "The A matrix is:" << "\n";
			print_matrix(A, A_size);
		}
		else {
			std::cout << "too big to print seed matrix and A matrix \n";
		}
	}

	//-------------------------------------------------------------------------------
	//Perform the cholesky decomposition and measure the execution time
	//-------------------------------------------------------------------------------

	std::cout << "Starting decomposition. \n";
	time_t begin, end; // time_t is a datatype to store time values.
	time(&begin); // note time before execution
	if (STABLE_FLAG) {
		stable_cholesky(A, D, L, 0, A_size, A_size);
		for (int col = 0; col < A_size; col++) {
			D[col] = sqrt(D[col]);
			for (int row = col; row < A_size; row++) {
				L[row][col] = L[row][col] * D[col];
			}
		}
	}
	else {
		cholesky(A, L, 0, A_size, A_size);
	}
	time(&end); // note time after execution
	//double execution_time = difftime(end, begin);

	//-------------------------------------------------------------------------------
	//Check to make sure that the 'seed' matrix equals the answer
	//-------------------------------------------------------------------------------

	if (check_equality(A_seed, L, A_size)) {
		std::cout << "Success! The decomposition completed in " << difftime(end, begin) << " seconds! \n";
	}
	else {
		std::cout << "Failure! The decomposition is incorrect. Printing the decomposition. \n";
		if (A_size < 200) {
			print_matrix(L, A_size);
			std::cout << "and the difference from seed is \n";
			print_difference(L, A_seed, A_size);
		}
		else {
			std::cout << "too big to print \n";
		}
	}
	
	for (int row = 0; row < A_size; ++row) {
		delete[] A_seed[row];
		delete[] A[row];
		delete[] L[row];
	}
	delete[] A;
	delete[] A_seed;
	delete[] L;
	delete[] D;

	system("pause");
    return 0;
}

void cholesky(double** A, double** L, int min_index, int max_index, int A_size) {
	if (DEBUG_FLAG) {
		std::cout << "Entering cholesky(), min_index = " << min_index << ", max_index = " << max_index << "\n";
	}
	if (max_index - min_index == 1){
		L[min_index][min_index] = sqrt(A[min_index][min_index]);
		if (DEBUG_FLAG) {
			std::cout << "Returning with L-matrix equal to: \n";
			print_matrix(L, A_size);
			system("pause");
		}
		return;
	}
	if (max_index - min_index == 2) {
		//the decomposition of 
		// a b
		// b c
		// is
		// sqrt(a) 0
		// b/sqrt(a) sqrt(c - b^2/a)
		L[min_index][min_index] = sqrt(A[min_index][min_index]);
		L[min_index + 1][min_index] = A[min_index+1][min_index] / L[min_index][min_index];
		L[min_index + 1][min_index + 1] = sqrt(A[min_index + 1][min_index + 1] - (L[min_index + 1][min_index] * L[min_index + 1][min_index]));
		if (DEBUG_FLAG) {
			std::cout << "Returning with L-matrix equal to: \n";
			print_matrix(L, A_size);
			system("pause");
		}
		return;
	}

	//---------------------------------------------------------------------------------------------------
	//Pick how to divide the matrix into blocks
	//---------------------------------------------------------------------------------------------------

	//int mid = std::min((min_index + max_index) / 2, min_index+400);
	int mid = std::max((min_index + max_index) / 2, max_index - 100);
	//int mid = (min_index + max_index) / 2;


	//---------------------------------------------------------------------------------------------------
	//Cholesky decompose the upper-right block
	//---------------------------------------------------------------------------------------------------
	cholesky(A, L, min_index, mid, A_size);

	//---------------------------------------------------------------------------------------------------
	//Solve for the elements of the lower left block using that
	//
	// if (A BT)  =  (L_A   0)  (L_A^T  L_B^T)
	//    (B D )  =  (L_B L_C)  (  0    L_C^T)
	// 
	// then B = L_B L_A^T
	//
	//---------------------------------------------------------------------------------------------------

	for (int col = min_index; col < mid; col++) {
		double inv_Lcolcol = 1 / L[col][col];
		for (int row = mid; row < max_index; row++) {
			L[row][col] = A[row][col];
			for (int c = min_index; c < col; c++) {
				L[row][col] -= L[row][c] * L[col][c];
			}
			L[row][col] = L[row][col] * inv_Lcolcol;
		}
	}

	if (DEBUG_FLAG) {
		std::cout << "Updated the corner region, L is now: \n";
		print_matrix(L, A_size);
		system("pause");
	}

	//---------------------------------------------------------------------------------------------------
	// Replace the bottom right block of A with 
	//
	//    (A BT) 
	//    (B D ) 
	// 
	// with D - B*BT
	//---------------------------------------------------------------------------------------------------

	for (int row = mid; row < max_index; row++) {
		for (int col = mid; col <= row; col++) {
			for (int c = min_index; c < mid; c++) {
				A[row][col] -= L[row][c] * L[col][c];
			}
		}
	}

	//---------------------------------------------------------------------------------------------------
	// Cholesky decompose the bottom right square
	//---------------------------------------------------------------------------------------------------

	cholesky(A, L, mid, max_index, A_size);

	//---------------------------------------------------------------------------------------------------
	// Return the bottom right block to its original values
	//---------------------------------------------------------------------------------------------------

	for (int row = mid; row < max_index; row++) {
		for (int col = mid; col <= row; col++) {
			for (int c = 0; c < mid; c++) {
				A[row][col] += L[row][c] * L[col][c];
			}
		}
	}
	return;
}


void stable_cholesky(double** A, double* D, double** L, int min_index, int max_index, int A_size) {
	if (DEBUG_FLAG) {
		std::cout << "Entering stable_cholesky(), min_index = " << min_index << ", max_index = " << max_index << "\n";
	}
	if (max_index - min_index == 1) {
		L[min_index][min_index] = 1;
		D[min_index] = A[min_index][min_index];
		if (DEBUG_FLAG) {
			std::cout << "Returning with L-matrix equal to: \n";
			print_matrix(L, A_size);
			std::cout << "And D-matrix \n";
			for (int col = 0; col < A_size; col++) { std::cout << D[col] << " "; }
			system("pause");
		}
		return;
	}
	if (max_index - min_index == 2) {
		//the stable decomposition of 
		// a b
		// b c
		// is
		// L = (1   0)
		//     (b/a 1)
		//
		// D = (a 0)
		//     (0 c-b^2/a)
		L[min_index][min_index] = 1;
		L[min_index + 1][min_index] = A[min_index + 1][min_index] / A[min_index][min_index];
		L[min_index + 1][min_index + 1] = 1;

		D[min_index] = A[min_index][min_index];
		D[min_index + 1] = A[min_index + 1][min_index + 1] - (L[min_index+1][min_index] * A[min_index+1][min_index]);
		if (DEBUG_FLAG) {
			std::cout << "Returning with L-matrix equal to: \n";
			print_matrix(L, A_size);
			std::cout << "And D-matrix \n";
			for (int col = 0; col < A_size; col++) { std::cout << D[col] << " "; }
			system("pause");
		}
		return;
	}

	//---------------------------------------------------------------------------------------------------
	//Pick how to divide the matrix into blocks
	//---------------------------------------------------------------------------------------------------

	//int mid = std::min((min_index + max_index) / 2, min_index+400);
	int mid = std::max((min_index + max_index) / 2, max_index - 100);
	//int mid = (min_index + max_index) / 2;


	//---------------------------------------------------------------------------------------------------
	//Cholesky decompose the upper-right block
	//---------------------------------------------------------------------------------------------------
	stable_cholesky(A, D, L, min_index, mid, A_size);

	//---------------------------------------------------------------------------------------------------
	//Solve for the elements of the lower left block using that
	//
	// if (A BT)  =  (L_A   0)  (D_A  0 )  (L_A^T  L_B^T)
	//    (B D )  =  (L_B L_C)  ( 0  D_C)  (  0    L_C^T)
	// 
	// then B = L_B D_A L_A^T
	//
	// To speed computation, 
	//      -first we calculate L_B D_A
	//      -then we replace the bottom right block of A (this uses L_B D_A) NO!!!
	//      -then we divide by the columns of L_B by the diagonal elements of D_A to get L_B
	//---------------------------------------------------------------------------------------------------

	for (int col = min_index; col < mid; col++) {
		for (int row = mid; row < max_index; row++) {
			L[row][col] = A[row][col];
			for (int c = min_index; c < col; c++) {
				L[row][col] -= L[row][c] * L[col][c];
			}
		}
	}
	for (int col = min_index; col < mid; col++) {
		double inverse_d = 1 / D[col];
		for (int row = mid; row < max_index; row++) {
			L[row][col] = L[row][col] * inverse_d;
		}
	}

	if (DEBUG_FLAG) {
		std::cout << "Updated the corner region, L is now: \n";
		print_matrix(L, A_size);
		std::cout << "And D-matrix \n";
		for (int col = 0; col < A_size; col++) { std::cout << D[col] << " "; }
		system("pause");
	}

	//---------------------------------------------------------------------------------------------------
	// Replace the bottom right block of A with 
	//
	//    (A BT) 
	//    (B C ) 
	// 
	// with C - L_B * D_A * L_B^T
	//
	//---------------------------------------------------------------------------------------------------

	for (int row = mid; row < max_index; row++) {
		for (int col = mid; col <= row; col++) {
			for (int c = min_index; c < mid; c++) {
				A[row][col] -= L[row][c] * D[c] * L[col][c];
			}
		}
	}

	//---------------------------------------------------------------------------------------------------
	// Cholesky decompose the bottom right square
	//---------------------------------------------------------------------------------------------------

	stable_cholesky(A, D, L, mid, max_index, A_size);

	//---------------------------------------------------------------------------------------------------
	// Return the bottom right block to its original values
	//---------------------------------------------------------------------------------------------------

	for (int row = mid; row < max_index; row++) {
		for (int col = mid; col <= row; col++) {
			for (int c = 0; c < mid; c++) {
				A[row][col] += L[row][c] * D[c] * L[col][c];
			}
		}
	}
	return;
}

void print_matrix(double** A, int A_size) {
	for (int row = 0; row < A_size; row++) {
		for (int col = 0; col <= row; col++) {
			std::cout << A[row][col] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	return;
}

void print_difference(double** A, double **B, int A_size) {
	for (int row = 0; row < A_size; row++) {
		for (int col = 0; col <= row; col++) {
			std::cout << A[row][col] - B[row][col] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	return;
}

void initialize_matrix_to_zero(double** A, int A_size) {
	for (int row = 0; row < A_size; row++) {
		for (int col = 0; col <= row; col++) {
			A[row][col] = 0; 
		}
	}
	return;
}


void initialize_seed(double** A_seed, int A_size) {
	for (int row = 0; row < A_size; row++) {
		for (int col = 0; col <= row; col++) {
			A_seed[row][col] = rand() % 5 + 1;
		}
	}
	return;
}

void A_from_seed(double** A, double** A_seed, int A_size) {
	for (int row = 0; row < A_size; row++) {
		for (int col = 0; col <= row; col++) {
			A[row][col] = 0;
			for (int c = 0; c <= col; c++) {
				A[row][col] += A_seed[row][c] * A_seed[col][c];
			}
		}
	}
}


bool check_equality(double** A, double** B, int A_size) {
	for (int row = 0; row < A_size; row++) {
		for (int col = 0; col <= row; col++) {
			if ((A[row][col] - B[row][col] > 0.00001) || (A[row][col] - B[row][col] < -0.00001)) {
				std::cout << "found a difference of size " << A[row][col] - B[row][col];
				return false;
			}
		}
	}
	return true;
}


