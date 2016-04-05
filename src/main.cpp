#include <gsl/gsl_vector.h>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include "cmdline.h"
#include "data.h"
#include "prob_functions.h"

void run(Dataset *data)
{
    int i, idx;
    const double THRESHOLD = 1E-6;
    const int maxiter = 15;
    double Q, lastQ;

    srand(time(NULL));

    // Initialize parameters to starting values
    if (data->initial_alpha_source.size() > 0) {
        std::cout << "Read initial alpha from "
                  << data->initial_alpha_source << std::endl;
        std::string line;
        std::string buffer;
        std::ifstream ifs(data->initial_alpha_source.c_str());
        if (ifs.fail()) {
            std::cerr << "Faild to open " << data->initial_alpha_source << std::endl;
            abort();
        }
        ifs >> line;  // Discard header
        while(getline(ifs, line)) {
            std::istringstream stream(line);
            int counter= -1;
            int idx = 0;
            while(getline(stream, buffer, ',')) {
                if (counter == -1)
                    idx = atoi(buffer.c_str());
                else
                    data->priorAlpha[idx] = atof(buffer.c_str());
                counter++;
            }
        }
        ifs.close();
    } else {
        for (i = 0; i < data->num_labelers; i++) {
            data->alpha[i] = data->priorAlpha[i];
            /*data->alpha[i] = (double) rand() / RAND_MAX * 4 - 2;*/
        }
    }
    for (idx = 0; idx < data->num_beta; idx++) {
        data->beta[idx] = data->priorBeta[idx];
        /*data->beta[j] = (double) rand() / RAND_MAX * 3;*/
    }
  
    EStep(data);
    Q = compute_objective_function(data);

    int iter = 0;
    do {
        lastQ = Q;

        // Estimate P(Z|L,alpha,beta)
        EStep(data);

        // Calculate the parameters (alpha, beta)
        MStep(data);

        Q = compute_objective_function(data);
        printf("[%d]: Q = %f\n", iter, Q);
        iter++;
        // if (Q < lastQ) Q = lastQ;
    } while (iter < maxiter && fabs((Q - lastQ)/lastQ) > THRESHOLD);
}

int main (int argc, char *argv[])
{
    Dataset data;

    cmdline::parser p;
    p.add<int>("mode", 'm',
               "mode flag with int value: 0..5", false,
               0, cmdline::range(1, 4));
    p.add<std::string>("prefix", 'p',
                       "prefix of output filenames (including dirname)",
                       false, "");
    p.add("ignore_prior_alpha", '\0',
          "ignore prior on alpha");
    p.add("ignore_prior_beta", '\0',
          "ignore prior on beta");
    p.add<double>("prior_alpha", '\0',
                  "initial values of prior on alpha",
                  false, 1.0);
    p.add<double>("prior_beta", '\0',
                  "initial values of prior on beta",
                  false, 1.0);
    p.add<double>("lambda_alpha", '\0',
                  "L2 regularization on alpha",
                  false, 0.0);
    p.add<double>("lambda_beta", '\0',
                  "L2 regularization on beta",
                  false, 0.0);
    p.add<std::string>("prior_z", '\0',
                       "initial values of prior on z",
                       false);
    p.add<std::string>("initial_alpha", '\0',
                       "input file to load initial value for alpha",
                       false);
    p.add("help", 0, "print help");
    p.add("debug", 'd', "debug mode");
    p.footer("<input> <hierarchy>");

    // help request or incorrect args
    if (!p.parse(argc, argv) || (p.rest().size() != 2)
        || p.exist("help")){
        std::cerr << p.error_full() << p.usage();
        return 0;
    }

    // set mode flag
    data.mode = p.get<int>("mode");

    // set flag if use prior for parameters
    data.ignore_priorAlpha = p.exist("ignore_prior_alpha");
    data.ignore_priorBeta = p.exist("ignore_prior_beta");

    // set initial values for alpha and beta
    data.init_priorAlpha = p.get<double>("prior_alpha");
    data.init_priorBeta = p.get<double>("prior_beta");

    data.lambdaAlpha = p.get<double>("lambda_alpha");
    data.lambdaBeta = p.get<double>("lambda_beta");

    data.initial_alpha_source = p.exist("initial_alpha") ?
        p.get<std::string>("initial_alpha") : "";
    data.debug = p.exist("debug") ? true : false;

    read_data(p.rest()[0], p.rest()[1], &data);

    run(&data);

    write_results(&data, p.get<std::string>("prefix"));
    free(data.labels);
    free(data.num_classes);
    free(data.num_children);
    free(data.steps);
    free(data.last_step);
    free(data.priorAlpha);
    free(data.priorBeta);
    free(data.alpha);
    free(data.beta);
    free(data.priorZ);
    free(data.probZ);
}
