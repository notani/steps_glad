#include <cmath>
#include "data.h"

void packX (gsl_vector *x, Dataset *data)
{
    int idx;

    // Pack parameters into a gsl_vector
    for (idx= 0; idx< data->num_labelers; idx++) {
        gsl_vector_set(x, idx, data->alpha[idx]);
    }
    for (idx = 0; idx < data->num_beta; idx++) {
        gsl_vector_set(x, data->num_labelers + idx, data->beta[idx]);
    }
}

void unpackX (const gsl_vector *x, Dataset *data)
{
    int idx;

    // Unpack parameters from a gsl_vector
    for (idx = 0; idx < data->num_labelers; idx++) {
        data->alpha[idx] = gsl_vector_get(x, idx);
    }
    for (idx = 0; idx < data->num_beta; idx++) {
        data->beta[idx] = gsl_vector_get(x, data->num_labelers + idx);
    }
}

// Set an index of intermediate class (value) at h-th tier for classId
void set_step(int classId, int h, int value, Dataset *data)
{
    data->steps[classId * (data->num_steps - 1) + h] = value;
}

// Returns an index of intermediate class at h-th tier for classId
// * Index starts from 0
int get_step(int classId, int h, Dataset *data) {
    if (h == data->last_step[classId] - 1)
        return classId;
    return data->steps[classId * (data->num_steps - 1) + h];
}

int get_rij(int l, int z, Dataset *data) {
    if (l == z)
        return data->last_step[z];

    int rij;
    for (rij = 0; rij < data->last_step[z] - 1; rij++)
        if (get_step(l, rij, data) != get_step(z, rij, data))
            break;
    return rij;
}

// Returns an index of the first class at h-th tier
int get_beta_index(int h, Dataset *data)
{
    int idx = 1;
    for (; h > 0; h--)
        idx += data->num_classes[h-1];
    return idx;
}

// Returns an index of class k at h-th tier
int get_beta_index(int h, int k, Dataset *data)
{
    return get_beta_index(h, data) + k;
}

// Returns an index of class k at h-th tier of item j
int get_beta_index(int j, int h, int k, Dataset *data)
{
    return j * data->num_beta_per_j + get_beta_index(h, k, data);
}

int get_z_index(int j, int k, Dataset *data)
{
    return j * data->num_leaves + k;
}

void get_beta_array(int z, double *beta, Dataset *data)
{
    int h, k = 0;
    for (h = 1; h < data->num_steps; h++) {
        k = get_step(z, h-1, data);
        beta[h] = data->beta[get_beta_index(h, k, data)];
    }
}

void get_num_diff_steps(const int z, int *count, Dataset *data)
{
    int h, offset;

    // Initialize counter
    for (h = 0; h <= data->num_steps; h++) 
        count[h] = 0;

    // Count
    count[0] = data->num_children[0] - 1;
    count[data->last_step[z]] = 1;
    offset = 1;
    for (h = 0; h < data->last_step[z] - 1; h++) {
        count[h + 1] = data->num_children[offset + get_step(z, h, data)] - 1;
        offset += data->num_classes[h];
    }
}

void get_num_diff_steps(const int z, int *count, int j, Dataset *data)
{
    int h, k;
    int offset = j * data->num_leaves;

    // Initialize counter
    for (h = 0; h <= data->num_steps; h++)
        count[h] = 0;

    // Count
    for (k = 0; k < data->num_leaves; k++) {
        if (k == z) {
            count[data->last_step[z]]++;
            continue;
        }
        if (data->priorZ[offset + k] == 0) {  // Ignore classes whose prior is 0
            continue;
        }
        for (h = 0; h < data->last_step[z] - 1; h++) {
            if (get_step(k, h, data) != get_step(z, h, data))
                break;
        }
        count[h] += 1;
    }
}

void read_steps(std::string filename, Dataset *data)
{
    int classId, value, last_step;
    int i, h, k;

    // Open file
    std::ifstream ifs(filename.c_str());
    if (ifs.fail()) {
        std::cerr << "Faild to open " << filename << std::endl;
        abort();
    }

    std::cout << "Read steps from " << filename << std::endl;

    // Initialize counter
    for (h = 0; h < data->num_steps - 1; h++)
        data->num_classes[h] = 0;

    // Read class hierarchies
    for (i = 0; i < data->num_leaves; i++) {
        ifs >> classId;
        last_step = -1;
        for (h = 0; h < data->num_steps - 1; h++) {
            ifs >> value;
            set_step(classId, h, value, data);  // Save parent's classId
            if (value == -1 && last_step == -1) {
                last_step = h + 1;
                continue;
            }
            if (data->num_classes[h] < value + 1) {  // Update #classes
                data->num_classes[h] = value + 1;
            }
        }
        // for the classes with full length of steps
        if (last_step == -1) {
            last_step = h + 1;
        }
        data->last_step[classId] = last_step;
    }
    // #classes at the last step
    data->num_classes[data->num_steps - 1] = 0;
    for (k = 0; k < data->num_leaves; k++) {
        if (data->last_step[k] != data->num_steps) continue;
        data->num_classes[data->num_steps - 1] += 1;
    }

    int numParents = 0;  // #parents
    int offset = 0;
    int parent, child;
    for (h = 0; h < data->num_steps - 1; h++) {
        numParents += data->num_classes[h];
    }
    std::cout << "numParents = " << numParents << std::endl;  // DEBUG
    std::vector< std::set<int> > counter(numParents);
    data->num_children = (int *) malloc(sizeof(int) * (1 + numParents));
    data->num_children[0] = data->num_classes[0];  // the 1st step
    for (h = 0; h < data->num_steps - 1; h++) {
        for (k = 0; k < data->num_leaves; k++) {
            if (data->last_step[k] <= h + 1) continue;
            parent = get_step(k, h, data);
            child = get_step(k, h + 1, data);
            counter[offset + parent].insert(child);  // Save child's classId
        }
        offset += data->num_classes[h];
    }

    for (i = 0; i < numParents; i++) {
        data->num_children[i + 1] = counter[i].size();
    }
    std::cout << "Finished reading hierarchy information. #Beta = (1,";
    for (h = 0; h < data->num_steps - 1; h++) {
        if (h != 0)
            std::cout << ",";
        std::cout << data->num_classes[h];
    }
    std::cout << ")" << std::endl;
    ifs.close();
}


void read_data (std::string filename, std::string filename_category, Dataset *data)
{
    int i, j, h, k, idx;
    std::ifstream ifs(filename.c_str());

    if (ifs.fail()) {
        std::cerr << "faild to open " << filename << std::endl;
        abort();
    }

    // Read parameters
    ifs >> data->num_labels >> data->num_labelers >> data->num_tasks >> data->num_leaves >> data->num_steps;
    // Allocate memory for probability and labels
    data->priorZ = (double *) malloc(sizeof(double) * data->num_tasks * data->num_leaves);
    data->probZ = (double *) malloc(sizeof(double) * data->num_tasks * data->num_leaves);
    data->labels = (Label *) malloc(sizeof(Label) * data->num_labels);
    // data->num_labels_i = (int *) malloc(sizeof(int) * data->num_labelers);

    // Initialize priorZ
    for (idx = 0; idx < data->num_tasks * data->num_leaves; idx++)
        data->priorZ[idx] = 1.0;

    // Read labels
    std::cout << "Read labels from " << filename << std::endl;
    for (idx = 0; idx < data->num_labels; idx++) {
        ifs >> data->labels[idx].imageIdx >> data->labels[idx].labelerId >> data->labels[idx].label;
        data->priorZ[data->labels[idx].imageIdx * data->num_leaves + data->labels[idx].label] = 1.0;
        // data->num_labels_i[data->labels[idx].labelerId]++;
    }

    // Close file
    ifs.close();

    // Read steps
    data->num_classes = (int *) malloc(sizeof(int) * data->num_steps);
    data->steps = (int *) malloc(sizeof(int) * data->num_leaves * (data->num_steps - 1));
    data->last_step = (int *) malloc(sizeof(int) * data->num_leaves);
    read_steps(filename_category, data);

    // Initialize
    std::cout << "#Labels = " << data->num_labels << std::endl;
    std::cout << "#Labelers = " << data->num_labelers << std::endl;
    std::cout << "#Items" << data->num_tasks << std::endl;
    std::cout << "#Steps" << data->num_steps << std::endl;

    switch (data->mode) {
    case 1:
    case 4:
        data->num_beta = data->num_tasks;
        break;
    case 2:
    case 3:
        data->num_beta = 1;  // parameter for root
        for (int h = 0; h < data->num_steps - 1; h++)
            data->num_beta += data->num_classes[h];
        if (data->mode == 3) {
            data->num_beta_per_j = data->num_beta;
            data->num_beta *= data->num_tasks;
        }
        break;
    default:
        std::cerr << "Invalid mode flag " << data->mode << std::endl;
        abort();
    }

    // Allocate memory for parameters
    data->priorAlpha = (double *) malloc(sizeof(double) * data->num_labelers);
    data->alpha = (double *) malloc(sizeof(double) * data->num_labelers);
    data->priorBeta = (double *) malloc(sizeof(double) * data->num_beta);
    data->beta = (double *) malloc(sizeof(double) * data->num_beta);

    // Initialize priorAlpha
    std::cout << "Assuming prior on alpha(1,..." << data->num_labelers << ") has mean "
              << data->init_priorAlpha << " and std 1.0"
              << std::endl;
    for (i = 0; i < data->num_labelers; i++)
        data->priorAlpha[i] = data->init_priorAlpha;  // default value for alpha

    // Initialize priorBeta
    std::cout << "Assuming prior on beta(1,...," << data->num_beta << ") has mean "
              << data->init_priorBeta << " and std 1.0"
              << std::endl;
    for (idx = 0; idx < data->num_beta; idx++)
        data->priorBeta[idx] = data->init_priorBeta;  // default value for beta

    // Initialize priorZ
    double *priorZ = (double *) malloc(sizeof(double) * data->num_leaves);
    int offset;
    for (k = 0; k < data->num_leaves; k++) {
        priorZ[k] = 1.0 / data->num_children[0];
        offset = 1;
        for (h = 0; h < data->last_step[k] - 1; h++) {
            priorZ[k] /= data->num_children[offset + get_step(k, h, data)];
            offset += data->num_classes[h];
        }
    }

    std::cout << "Assuming p(Z=1) is unique according to the tree structure" << std::endl;
    for (j = 0; j < data->num_tasks; j++) {
        for (k = 0; k < data->num_leaves; k++) {
            data->priorZ[j * data->num_leaves + k] = priorZ[k];
        }
    }
}


void write_results(Dataset *data, std::string prefix)
{
    std::ofstream ofs;
    int i, j, h, k, jh;
    int offset;
    std::string filename;
    filename = prefix + "_alpha.csv";
    ofs.open(filename.c_str());
    ofs << "id,val" << std::endl;
    for (i = 0; i < data->num_labelers; i++) {
        ofs << i << "," << std::fixed << data->alpha[i] << std::endl;
    }
    ofs.close();

    filename = prefix + "_beta.csv";
    ofs.open(filename.c_str());
    ofs << "id,val" << std::endl;
    switch (data->mode) {
    case 2:
    case 5:
        for (h = 0; h < data->num_steps; h++) {
            ofs << h << "," << std::fixed << exp(data->beta[h]) << std::endl;
        }
        break;
    case 6:
        ofs << "0-0," << data->beta[0] << std::endl;
        offset = 1;
        for (h = 1; h < data->num_steps; h++) {
            for (k = 0; k < data->num_classes[h-1]; k++) {
                ofs << h << "-" << k << "," << data->beta[offset + k]
                    << std::endl;
            }
            offset += data->num_classes[h-1];
        }
        break;
    default:
        for (j = 0; j < data->num_tasks; j++) {
            switch (data->mode) {
            case 0:
                ofs << j << ",";
                for (h = 0; h < data->num_steps; h++) {
                    jh = j * data->num_steps + h;
                    if (h != 0)
                        ofs << ",";
                    ofs << std::fixed << exp(data->beta[jh]);
                }
                ofs << std::endl;
                break;
            case 3:
                ofs << j << ",";
                for (h = 0; h < data->num_steps; h++) {
                    jh = j * data->num_steps + h;
                    if (h != 0)
                        ofs << ",";
                    ofs << std::fixed << data->beta[jh];
                }
                ofs << std::endl;
                break;
            case 1:
            case 4:
                ofs << j << "," << std::fixed << exp(data->beta[j]) << std::endl;
                break;
            case 7:
            case 8:
                ofs << j << ",";
                ofs << "0-0," << data->beta[0] << std::endl;
                offset = 1;
                for (h = 1; h < data->num_steps; h++) {
                    ofs << j << ",";
                    for (k = 0; k < data->num_classes[h-1]; k++) {
                        ofs << h << "-" << k << "," << data->beta[offset + k]
                            << std::endl;
                    }
                    offset += data->num_classes[h-1];
                }
                break;
            default:
                std::cerr << "Invalid mode flag " << data->mode << std::endl;
                abort();
            }
        }
        break;
    }
    ofs.close();

    filename = prefix + "_probs.csv";
    ofs.open(filename.c_str());
    ofs << "id";
    for (k = 0; k < data->num_leaves; k++)
        ofs << "," << k;
    ofs << std::endl;
    for (j = 0; j < data->num_tasks; j++) {
        ofs << j;
        for (k = 0; k < data->num_leaves; k++) {
            ofs << "," << std::fixed << data->probZ[j * data->num_leaves + k];
        }
        ofs << std::endl;
    }
    ofs.close();
}
