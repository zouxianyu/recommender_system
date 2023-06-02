#ifndef RECOMMENDER_SYSTEM_CORE_HPP
#define RECOMMENDER_SYSTEM_CORE_HPP

#include <string>
#include "sparse_matrix.hpp"

constexpr int FEAT_USE_ATTR = 1;
constexpr int FEAT_USE_WEIGHT = 2;

SparseMatrix<double> read_train_dataset(const std::string &filename);

SparseMatrix<double> read_test_dataset(const std::string &filename);

SparseMatrix<int> read_item_attribute(const std::string &filename);

void write_dataset(const std::string &filename,
                   const SparseMatrix<double> &mat);

void write_dataset_in_order(const std::string &reference,
                            const std::string &filename,
                            const SparseMatrix<double> &mat);

std::pair<SparseMatrix<double>, SparseMatrix<double>> make_train_test(
        const SparseMatrix<double> &mat, size_t test_count);

SparseMatrix<double> predict(const SparseMatrix<double> &user_mat,
                             const SparseMatrix<double> &test_user_mat,
                             const SparseMatrix<int> &item_attr,
                             int k,
                             int flags);

double RMSE(const SparseMatrix<double> &mat1,
            const SparseMatrix<double> &mat2);

#endif //RECOMMENDER_SYSTEM_CORE_HPP
