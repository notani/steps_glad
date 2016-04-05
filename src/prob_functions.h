#ifndef PROB_FUNCTIONS_H
#define PROB_FUNCTIONS_H

#include <gsl/gsl_vector.h>
#include "data.h"

// EM algorithm
void EStep(Dataset *data);
void MStep(Dataset *data);

// Q function for the EM algorithm
double compute_objective_function(Dataset *data);

// Steps GLAD (task-dependent)
double calc_log_ProbL_GLAD_t(int, int, int, double, double, int);
// Steps GLAD (class & task-and-class dependent)
double calc_log_ProbL_GLAD_ctc(int, int, double, double*, int*, int);
// Steps Rasch model (task-dependent), presented as an extention example
double calc_log_ProbL_rasch_t(int, int, int, double, double, int);

double sigmoid(double x);
double log_sigmoid(double x);

// Compute gradients for the M step
void compute_gradients(Dataset*, double*, double*);
// Steps GLAD (task-dependent)
void roundAlpha_GLAD_t(double*, int, int, double, double, double);
void roundBeta_GLAD_t(double*, int, int, double, double, double, double);
// Steps GLAD (class & task-and-class dependent)
void roundAlpha_GLAD_ctc(double*, int, int, double, double, double*, int*);
void roundBeta_GLAD_ctc(double*, int, int, double, double, double*, int*);
// Steps Rasch model (task-dependent), presented as an extention example
void roundAlpha_rasch_t(double*, int, int, double, double);
void roundBeta_rasch_t(double*, int, int, double, double, double);

// GSL minimizer
double objective_function(const gsl_vector*, void*);
void get_gradients(const gsl_vector*, void*, gsl_vector*);
void set_functions(const gsl_vector*, void*, double*, gsl_vector*);
#endif
