#include <iostream>
#include <math.h>
#include <string.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sf_erf.h>
#include "prob_functions.h"
#include "data.h"

void EStep (Dataset *data)
{
    int i, j, k, lij, rij, idx;
    double p, denom;
    int *count = (int *) malloc(sizeof(int) * (data->num_steps + 1));
    int *beta_idx = (int *) malloc(sizeof(int) * data->num_steps);
    beta_idx[0] = 0;

    if (data->debug) {
        std::cerr << "EStep" << std::endl;
    }

    // prior Z
    for (j = 0; j < data->num_tasks; j++) {
        for (k = 0; k < data->num_leaves; k++) {
            idx = get_z_index(j, k, data);
            data->probZ[idx] = log(data->priorZ[idx]);
        }
    }

    for (k = 0; k < data->num_leaves; k++) {  // true class
        // class-dependent beta
        if (data->mode == 2) {
            for (int h = 0; h < data->last_step[k] - 1; h++) {
                beta_idx[h+1] = get_beta_index(h, get_step(k, h, data), data);
            }
        }
        for (idx = 0; idx < data->num_labels; idx++) {
            i = data->labels[idx].labelerId;
            j = data->labels[idx].imageIdx;
            lij = data->labels[idx].label;

            // task-and-class dependent beta
            if (data->mode == 3) {
                for (int h = 0; h < data->last_step[k] - 1; h++) {
                    beta_idx[h+1] = get_beta_index(j, h, get_step(k, h, data), data);
                }
            }

            // Find rij
            get_num_diff_steps(k, count, j, data);
            rij = get_rij(lij, k, data);

            // Compute logP(l); the probability of generating the label
            switch (data->mode) {
            case 1:  // Steps GLAD (task-dependent)
                p = calc_log_ProbL_GLAD_t(lij, rij,
                                          data->last_step[k],
                                          data->alpha[i],
                                          data->beta[j],
                                          count[rij]);
                break;
            case 2:  // Steps GLAD (class dependent)
            case 3:  // Steps GLAD (task-and-class dependent)
                p = calc_log_ProbL_GLAD_ctc(rij,
                                            data->last_step[k],
                                            data->alpha[i],
                                            data->beta, beta_idx,
                                            count[rij]);
                break;
            case 4:  // Steps Rasch model (task dependent)
                // This is presented as another example of extenstion
                p = calc_log_ProbL_rasch_t(lij, rij,
                                           data->last_step[k],
                                           data->alpha[i],
                                           data->beta[j],
                                           count[rij]);
                break;
            default:
                std::cerr << "Invalid mode " << data->mode << std::endl;
                abort();
            }

            data->probZ[get_z_index(j, k, data)] += p;
        }
    }

    // Exponentiate and Normalize
    for (j = 0; j < data->num_tasks; j++) {
        denom = 0.0;
        for (k = 0; k < data->num_leaves; k++) {
            idx = get_z_index(j, k, data);
            if (data->probZ[idx] == -INFINITY) {
                data->probZ[idx] = 0;
                continue;
            }
            data->probZ[idx] = exp(data->probZ[idx]);
            denom += data->probZ[idx];
        }

        for (k = 0; k < data->num_leaves; k++) {
            idx = get_z_index(j, k, data);
            data->probZ[idx] /= denom;
            if (isnan(data->probZ[idx])) {
                std::cerr << "ERROR: isnan(data->probZ[idx]) [EStep]" << std::endl;
                std::cerr << denom << std::endl;
                abort();
            }
        }
    }
    free(count);
    free(beta_idx);
}

void MStep (Dataset *data)
{
    double lastF;
    int iter, status;
    int numParams = data->num_labelers + data->num_beta;

    gsl_vector *x;
    gsl_multimin_fdfminimizer *s;
    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_function_fdf obj_func;

    gsl_set_error_handler_off();

    // Pack parameters into a gsl_vector
    x = gsl_vector_alloc(numParams);
    packX(x, data);

    std::cerr << "Packed" << std::endl;

    // Initialize objective function
    obj_func.n = numParams;
    obj_func.f = &objective_function;
    obj_func.df = &get_gradients;
    obj_func.fdf = &set_functions;
    obj_func.params = data;
    // Initialize a minimizer
    T = gsl_multimin_fdfminimizer_conjugate_pr;
    s = gsl_multimin_fdfminimizer_alloc(T, numParams);

    gsl_multimin_fdfminimizer_set(s, &obj_func, x, 0.01, 0.01);
    iter = 0;

    std::cerr << "Start optimization" << std::endl;
    do {
        lastF = s->f;
        iter++;
        printf("[MStep:%d] f=%f\n", iter, s->f);

        status = gsl_multimin_fdfminimizer_iterate(s);
        if (status) {
            printf("Error occurred [MStep]\n");
            break;
        }
        status = gsl_multimin_test_gradient(s->gradient, 1e-2);
        if (status == GSL_SUCCESS) {
            printf ("Minimum found");
        }
    } while (lastF - s->f > 1e-5 && status == GSL_CONTINUE && iter < 25);

    /* Unpack alpha and beta from gsl_vector */
    unpackX(s->x, data);
  
    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}

double compute_objective_function(Dataset *data)
{
    int i, j, k, idx, lij, rij;
    int *count = (int *) malloc(sizeof(int) * (data->num_steps + 1));
    double p;
    double Q = 0;

    double *alpha = data->alpha, *beta = data->beta;
    int *beta_idx = (int *) malloc(sizeof(int) * data->num_steps);
    beta_idx[0] = 0;

    // Start with the expectation of the sum of priors over all images
    for (j = 0; j < data->num_tasks; j++) {
        for (k = 0; k < data->num_leaves; k++) {
            idx = get_z_index(j, k, data);
            if (data->priorZ[idx] == 0) continue;  // Skip ignored Z
            Q += data->probZ[idx] * log(data->priorZ[idx]);
        }
    }

    for (k = 0; k < data->num_leaves; k++) {  // True class
        // class-dependent beta
        if (data->mode == 2) {
            for (int h = 0; h < data->last_step[k] - 1; h++)
                beta_idx[h+1] = get_beta_index(h, get_step(k, h, data), data);
        }

        for (idx = 0; idx < data->num_labels; idx++) {
            i = data->labels[idx].labelerId;
            j = data->labels[idx].imageIdx;
            lij = data->labels[idx].label;

            // task-and-class dependent beta
            if (data->mode == 3) {
                for (int h = 0; h < data->last_step[k] - 1; h++)
                    beta_idx[h+1] = get_beta_index(j, h, get_step(k, h, data), data);
            }

            // Find rij
            get_num_diff_steps(k, count, j, data);
            rij = get_rij(lij, k, data);

            // Compute logP(l)
            switch (data->mode) {
            case 1:  // Steps GLAD (task-dependent)
                p = calc_log_ProbL_GLAD_t(lij, rij,
                                          data->last_step[k],
                                          data->alpha[i],
                                          data->beta[j],
                                          count[rij]);
                break;
            case 2:  // Steps GLAD (class dependent)
            case 3:  // Steps GLAD (task-and-class dependent)
                p = calc_log_ProbL_GLAD_ctc(rij,
                                            data->last_step[k],
                                            data->alpha[i],
                                            data->beta, beta_idx,
                                            count[rij]);
                break;
            case 4:  // Steps Rasch model (task dependent)
                // This is presented as another example of extenstion
                p = calc_log_ProbL_rasch_t(lij, rij,
                                           data->last_step[k],
                                           data->alpha[i],
                                           data->beta[j],
                                           count[rij]);
                break;
            default:
                std::cerr << "Invalid mode " << data->mode << std::endl;
                abort();
            }

            Q += p * data->probZ[get_z_index(j, k, data)];
        }
        if (isnan(Q)) {
            std::cerr << "isnan(Q) is True after computing Q from labels: Q = " << Q << std::endl;
            abort();
        }
    }

    // Reguralization penalty (default: lambda = 0)
    for (i = 0; i < data->num_labelers; i++) {
        Q -= data->lambdaAlpha
            * (data->alpha[i] - data->priorAlpha[i])
            * (data->alpha[i] - data->priorAlpha[i]);
    }
    for (idx = 0; idx < data->num_beta; idx++) {
        Q -= data->lambdaBeta
            * (data->beta[idx] - data->priorBeta[idx])
            * (data->beta[idx] - data->priorBeta[idx]);
    }

    // Add Gaussian (standard normal) prior for alpha
    if (!data->ignore_priorAlpha) {
        for (i = 0; i < data->num_labelers; i++) {
            Q += log(gsl_sf_erf_Z(alpha[i] - data->priorAlpha[i]));
            if (isnan(Q)) {
                std::cerr << "isnan(Q) is True after adding Gaussian prior for alpha" << std::endl;
                abort();
            }
        }
    }

    // Add Gaussian (standard normal) prior for beta
    if (!data->ignore_priorBeta) {
        for (idx = 0; idx < data->num_beta; idx++) {
            Q += log(gsl_sf_erf_Z(beta[idx] - data->priorBeta[idx]));
            if (isnan(Q)) {
                std::cerr << "isnan(Q) is after True adding Gaussian prior for beta" << std::endl;
                abort();
            }
        }
    }

    if (data->debug) {
        std::cerr << "Q = " << Q << std::endl;
    }
    free(count);
    free(beta_idx);

    return Q;
}

// Steps GLAD (task-dependent)
double calc_log_ProbL_GLAD_t(int lij, int rij, int s,
                             double alpha, double beta, int K)
{
    int upper = (s < (rij + 1)) ? s : (rij + 1);
    double p;
    p = rij * exp(beta) * alpha;
    p += upper * log_sigmoid(- exp(beta) * alpha);
    p -= log(K);
    return p;
}

// Steps GLAD (class & task-and-class dependent)
double calc_log_ProbL_GLAD_ctc(int rij, int s,
                               double alpha, double *beta, int *beta_idx, int K)
{
    int h;
    int upper = (s < (rij + 1)) ? s : (rij + 1);
    double p = 0.0;

    for (h = 0; h < rij; h++)
        p += exp(beta[beta_idx[h]]);
    p *= alpha;
    for (h = 0; h < upper; h++)
        p += log_sigmoid(- exp(beta[beta_idx[h]]) * alpha);
    p -= log(K);
    return p;
}

// Steps Rasch model (task dependent)
// This is presented as another example of extenstion
double calc_log_ProbL_rasch_t(int lij, int rij, int s,
                              double alpha, double beta, int K)
{
    int upper = (s < (rij + 1)) ? s : (rij + 1);
    return - rij * (beta - alpha) + upper * log_sigmoid(beta - alpha) - log(K);
}

double sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
}

double log_sigmoid(double x)
{
    double v = -log(1 + exp(-x));
    // Do some analytic manipulation first for numerical stability!
    if (v == -INFINITY) {
        std::cout << "approximation [log_sigmoid]" << std::endl;
        v = x;  // For large negative x, -log(1 + exp(-x)) = x
    }
    return v;
}

//----------------------------------------------------------------------
// Gradients
//----------------------------------------------------------------------
void compute_gradients(Dataset *data, double *dQdAlpha, double *dQdBeta)
{
    int i, j, h, k, lij, rij = 0;
    int idx, upper;
    double sigma;
    int *beta_idx = (int *) malloc(sizeof(int) * data->num_steps);
    beta_idx[0] = 0;

    // This comes from the priors
    for (i = 0; i < data->num_labelers; i++) {
        dQdAlpha[i] = (data->ignore_priorAlpha) ? 0 : - (data->alpha[i] - data->priorAlpha[i]);
        // Regularization Penalty (default: lambda = 0)
        dQdAlpha[i] = - 2.0 * (data->alpha[i] - data->priorAlpha[i]);
    }

    for (idx = 0; idx < data->num_beta; idx++) {
        dQdBeta[idx] = (data->ignore_priorBeta) ? 0 : - (data->beta[idx] - data->priorBeta[idx]);
        // Regularization Penalty (default: lambda = 0)
        dQdBeta[idx] = - 2 * data->lambdaBeta * (data->beta[idx] - data->priorBeta[idx]);
    }

    i = j = 0;
    for (k = 0; k < data->num_leaves; k++) {  // True class
        // class-dependent beta
        if (data->mode == 2) {
            for (h = 0; h < data->last_step[k] - 1; h++) {
                beta_idx[h+1] = get_beta_index(h, get_step(k, h, data), data);
            }
        }
        for (idx = 0; idx < data->num_labels; idx++) {
            i = data->labels[idx].labelerId;
            j = data->labels[idx].imageIdx;
            lij = data->labels[idx].label;

            // task-and-class dependent beta
            if (data->mode == 3) {
                for (h = 0; h < data->last_step[k] - 1; h++) {
                    beta_idx[h+1] = get_beta_index(j, h, get_step(k, h, data), data);
                }
            }

            // Find rij
            rij = get_rij(lij, k, data);

            upper = (data->last_step[k] < (rij + 1)) ? data->last_step[k] : (rij + 1);

            // Calculate dQdAlpha and dQdBeta
            switch (data->mode) {
            case 1:
                sigma = sigmoid(exp(data->beta[j]) * data->alpha[i]);
                roundAlpha_GLAD_t(&dQdAlpha[i], rij, upper,
                                  data->probZ[j * data->num_leaves + k],
                                  sigma, data->beta[j]);
                roundBeta_GLAD_t(&dQdBeta[j], rij, upper,
                                 data->probZ[j * data->num_leaves + k],
                                 sigma, data->alpha[i], data->beta[j]);
                break;
            case 2:
            case 3:
                roundAlpha_GLAD_ctc(&dQdAlpha[i], rij, upper,
                                    data->probZ[get_z_index(j, k, data)],
                                    data->alpha[i], data->beta, beta_idx);
                roundBeta_GLAD_ctc(dQdBeta, rij, upper,
                                   data->probZ[get_z_index(j, k, data)],
                                   data->alpha[i], data->beta, beta_idx);
                break;
            case 4:
                sigma = sigmoid(data->alpha[i] - data->beta[j]);
                roundAlpha_rasch_t(&dQdAlpha[i], rij, upper,
                                   data->probZ[j * data->num_leaves + k],
                                   sigma);
                roundBeta_rasch_t(&dQdBeta[j], rij, upper,
                                  data->probZ[j * data->num_leaves + k],
                                  sigma, data->beta[j]);
                break;
            default:
                std::cerr << "Invalid mode flag " << data->mode << std::endl;
                abort();
            }
        }
    }

    if (data->debug) {
        std::cerr << "da[7] = " << dQdAlpha[7] << " ";
        std::cerr << "da[52] = " << dQdAlpha[52] << " ";
        std::cerr << "da[74] = " << dQdAlpha[74] << std::endl;
        std::cerr << "db[0] = " << dQdBeta[0] << " ";
        std::cerr << "db[1] = " << dQdBeta[1] << std::endl;
    }

    free(beta_idx);
}

// Steps GLAD (task-dependent)    
void roundAlpha_GLAD_t(double *dQdAlpha, int rij, int upper,
                       double p, double sigma, double beta)
{
    *dQdAlpha += p * (rij - upper * sigma) * exp(beta);
}

void roundBeta_GLAD_t(double *dQdBeta, int rij, int upper,
                     double p, double sigma, double alpha, double beta)
{
    *dQdBeta += p * (rij - upper * sigma) * alpha * exp(beta);
}

// Steps GLAD (class & task-and-class dependent)
void roundAlpha_GLAD_ctc(double *dQdAlpha, int rij, int upper,
                         double p, double alpha, double *beta, int *beta_idx)
{
    int h, idx;
    double sum = 0.0;
    for (h = 0; h < rij; h++)
        sum += exp(beta[beta_idx[h]]);
    for (h = 0; h < upper; h++) {
        idx = beta_idx[h];
        sum -= exp(beta[idx]) * sigmoid(exp(beta[idx]) * alpha);
    }
    *dQdAlpha += p * sum;
}

void roundBeta_GLAD_ctc(double *dQdBeta, int rij, int upper,
                        double p, double alpha, double *beta, int *beta_idx)
{
    int h, idx;
    double sum = 0.0;
    for (h = 0; h < upper; h++) {
        idx = beta_idx[h];
        sum = 0.0;
        if (rij > h)
            sum += 1.0;
        if (rij + 1 > h)
            sum -= 1.0 * sigmoid(exp(beta[idx]) * alpha);
        dQdBeta[idx] += p * sum * alpha * exp(beta[idx]);
    }
}

// Steps Rasch model (task-dependent), presented as an extention example
void roundAlpha_rasch_t(double *dQdAlpha, int rij, int upper,
                        double p, double sigma)
{
    *dQdAlpha += p * (rij - upper * sigma);
}

void roundBeta_rasch_t(double *dQdBeta, int rij, int upper,
                       double p, double sigma, double beta)
{
    *dQdBeta += p * (- rij + upper * sigma);
}


//----------------------------------------------------------------------
// GSL minimization
//----------------------------------------------------------------------
// Function to be minimized with GSL minimizer
double objective_function(const gsl_vector *x, void *params)
{
    Dataset *data = (Dataset *) params;

    unpackX(x, data);

    return -compute_objective_function(data);
}

// Set gradients into a gsl vector
void get_gradients(const gsl_vector *x, void *params, gsl_vector *g)
{
    int i, idx;

    Dataset *data = (Dataset *) params;
    double *dQdAlpha = (double *) malloc(sizeof(double) * data->num_labelers);
    double *dQdBeta = (double *) malloc(sizeof(double) * data->num_beta);

    unpackX(x, data);

    compute_gradients(data, dQdAlpha, dQdBeta);

    /* Pack dQdAlpha and dQdBeta into gsl_vector */
    for (i = 0; i < data->num_labelers; i++) {
        gsl_vector_set(g, i, - dQdAlpha[i]);  /* Flip the sign since we want to minimize */
    }
    for (idx = 0; idx < data->num_beta; idx++) {
        gsl_vector_set(g, data->num_labelers + idx, - dQdBeta[idx]);  /* Flip the sign since we want to minimize */
    }
    free(dQdAlpha);
    free(dQdBeta);
}

// Set the objective function and gradients for GSL minimizer
void set_functions(const gsl_vector *x, void *params, double *f, gsl_vector *g)
{
    *f = objective_function(x, params);
    get_gradients(x, params, g);
}
