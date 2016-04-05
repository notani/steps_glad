#ifndef DATA_H
#define DATA_H

#include <gsl/gsl_vector.h>
#include <iostream>
#include <fstream>
#include <set>
#include <string>
#include <vector>

typedef struct {
    int imageIdx;
    int labelerId;
    int label;
} Label;

typedef struct {
    Label *labels;
    int *steps; // the valus of each step linked with the value of the last step
    int *last_step; // the valus of each step linked with the value of the last step
    int num_labels;
    int num_labelers;
    int num_tasks;
    int num_steps;
    int *num_classes, *num_children;
    int num_leaves;
    int num_beta;
    int num_beta_per_j;
    double *priorAlpha, *priorBeta;
    double init_priorAlpha, init_priorBeta;
    double *alpha, *beta;
    double *priorZ;  // 2 dimensional array expressed as 1 dimensional
    double *probZ;  // 2 dimensional array expressed as 1 dimensional
    double lambdaAlpha, lambdaBeta;  // L2 regulatiozation penalty
    std::string initial_alpha_source;

    // Flags
    int mode;  // 1=task-dep., 2=class-dep., 3=task&class-dep.
    bool debug;
    bool ignore_priorAlpha, ignore_priorBeta;
} Dataset;

void packX(gsl_vector *, Dataset *);
void unpackX (const gsl_vector *, Dataset*);
void set_step(int, int, int, Dataset*);
int get_step(int, int, Dataset*);
int get_rij(int, int, Dataset*);
int get_beta_index(int, Dataset*);
int get_beta_index(int, int, Dataset*);
int get_beta_index(int, int, int, Dataset*);
int get_z_index(int, int, Dataset*);
void get_beta_array(int, double*, Dataset*);
void get_num_diff_steps(const int, int*, Dataset*);
void get_num_diff_steps(const int, int*, int, Dataset*);
void read_steps(std::string, Dataset*);
void read_data(std::string, std::string, Dataset*);
void outputResults (Dataset*);
void write_results(Dataset*, std::string);
#endif
