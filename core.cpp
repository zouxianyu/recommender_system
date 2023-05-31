#include <fstream>
#include <stdexcept>
#include <map>
#include <algorithm>
#include <iostream>
#include <vector>
#include <indicators/progress_bar.hpp>
#include "core.hpp"

using namespace indicators;

using Item = SparseMatrix<double>::Item;

SparseMatrix<double> read_dataset(const std::string &filename, bool has_score) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    std::vector<Item> items;
    char split;
    size_t user_id, items_count;
    while (!file.eof() &&
           file >> user_id >> split >> items_count) {
        for (size_t i = 0; i < items_count; ++i) {
            size_t item_id;
            double score;
            file >> item_id;
            if (has_score) {
                file >> score;
            } else {
                score = 0;
            }
            items.emplace_back(user_id, item_id, score);
        }
    }
    return SparseMatrix<double>(items);
}

void write_dataset(const std::string &filename,
                   const SparseMatrix<double> &mat) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }

    for (size_t row_id: mat.row_indexes()) {
        std::span<const Item> row = mat.get_row(row_id);
        file << row_id << "|" << row.size() << std::endl;
        for (const auto &item: row) {
            file << item.col << "  " << item.val << std::endl;
        }
    }
}

SparseMatrix<double> read_train_dataset(const std::string &filename) {
    return read_dataset(filename, true);
}

SparseMatrix<double> read_test_dataset(const std::string &filename) {
    return read_dataset(filename, false);
}

std::pair<SparseMatrix<double>, SparseMatrix<double>> make_train_test(
        const SparseMatrix<double> &mat, size_t test_count) {
    std::vector<Item> train_items;
    std::vector<Item> test_items;
    for (size_t row_id: mat.row_indexes()) {
        std::span<const Item> row = mat.get_row(row_id);
        if (row.size() <= test_count) {
            continue;
        }
        auto train_row = row.subspan(0, row.size() - test_count);
        auto test_row = row.subspan(row.size() - test_count, test_count);
        train_items.insert(train_items.end(), train_row.begin(),
                           train_row.end());
        test_items.insert(test_items.end(), test_row.begin(), test_row.end());
    }
    return {SparseMatrix<double>(train_items),
            SparseMatrix<double>(test_items)};
}

std::map<size_t, double> get_avg_score_by_row(const SparseMatrix<double> &mat) {
    std::map<size_t, double> avg_score;
    for (const auto &row_id: mat.row_indexes()) {
        double sum = 0;
        size_t count = 0;
        for (const auto &item: mat.get_row(row_id)) {
            sum += item.val;
            ++count;
        }
        avg_score[row_id] = sum / count;
    }
    return avg_score;
}

double get_global_avg_score(const SparseMatrix<double> &mat) {
    double sum = 0;
    size_t count = 0;
    for (const auto &item: mat.get_all()) {
        sum += item.val;
        ++count;
    }
    return sum / count;
}

template<typename T>
inline T square(T x) { return x * x; }

double pearson(const SparseMatrix<double> &mat, size_t x, size_t y,
               const std::map<size_t, double> &avg_score) {
    std::span<const Item> row_x = mat.get_row(x);
    std::span<const Item> row_y = mat.get_row(y);
    double avg_x = avg_score.at(x);
    double avg_y = avg_score.at(y);

    size_t i = 0;
    size_t j = 0;
    double numerator = 0;
    double denominator_x = 0;
    double denominator_y = 0;
    while (i < row_x.size() && j < row_y.size()) {
        if (row_x[i].col < row_y[j].col) {
            denominator_x += square(row_x[i].val - avg_x);
            ++i;
        } else if (row_x[i].col > row_y[j].col) {
            denominator_y += square(row_y[j].val - avg_y);
            ++j;
        } else {
            numerator += (row_x[i].val - avg_x) * (row_y[j].val - avg_y);
            denominator_x += square(row_x[i].val - avg_x);
            denominator_y += square(row_y[j].val - avg_y);
            ++i;
            ++j;
        }
    }
    if (i != row_x.size()) {
        while (i < row_x.size()) {
            denominator_x += square(row_x[i].val - avg_x);
            ++i;
        }
    }
    if (j != row_y.size()) {
        while (j < row_y.size()) {
            denominator_y += square(row_y[j].val - avg_y);
            ++j;
        }
    }
    double denominator = std::sqrt(denominator_x * denominator_y);
    if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
        return 0;
    }
    return numerator / denominator;
}

bool heap_compare(const std::pair<size_t, double> &a,
                  const std::pair<size_t, double> &b) {
    return a.second > b.second;
}

void update_top_k_score(std::vector<std::pair<size_t, double>> &top_k,
                        size_t k, size_t id, double score) {
    if (top_k.size() < k) {
        top_k.emplace_back(id, score);
        std::push_heap(top_k.begin(), top_k.end(), heap_compare);
    } else if (top_k.front().second < score) {
        std::pop_heap(top_k.begin(), top_k.end(), heap_compare);
        top_k.back() = {id, score};
        std::push_heap(top_k.begin(), top_k.end(), heap_compare);
    }
}

std::map<size_t, std::vector<std::pair<size_t, double>>> get_top_k_similar_mat(
        const SparseMatrix<double> &mat, size_t k,
        const std::map<size_t, double> &avg_score) {

    std::map<size_t, std::vector<std::pair<size_t, double>>> result;

    std::vector<size_t> row_ids =
            {mat.row_indexes().begin(), mat.row_indexes().end()};

    for (size_t i: row_ids) {
        std::vector<std::pair<size_t, double>> empty;
        empty.reserve(k);
        result[i] = empty;
    }

    // info for progress bar
    const size_t all_count = row_ids.size() * (row_ids.size() - 1) / 2;
    size_t current_count = 0;
    ProgressBar bar{
            option::PrefixText{"Calculating Pearson Score"},
            option::BarWidth{50},
            option::ShowPercentage{true},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
    };

    for (size_t i = 0; i < row_ids.size(); ++i) {
        for (size_t j = i + 1; j < row_ids.size(); ++j) {
            size_t x = row_ids[i];
            size_t y = row_ids[j];
            auto &result_x = result[x];
            auto &result_y = result[y];
            double score = pearson(mat, x, y, avg_score);
            update_top_k_score(result_x, k, y, score);
            update_top_k_score(result_y, k, x, score);

            // show progress bar
            double progress = static_cast<double>(++current_count) / all_count;
            if (current_count == all_count || current_count % 1000000 == 0) {
                bar.set_progress(progress * 100);
            }
        }
    }

    for (size_t i: row_ids) {
        auto &heap = result[i];
        std::sort_heap(heap.begin(), heap.end(), heap_compare);
        std::reverse(heap.begin(), heap.end());
    }

    return result;
}

SparseMatrix<double> solve(const SparseMatrix<double> &user_mat,
                           const SparseMatrix<double> &test_user_mat) {
    SparseMatrix<double> item_mat = user_mat.transpose();

    double global_avg_score = get_global_avg_score(user_mat);
    std::map<size_t, double> user_avg_score = get_avg_score_by_row(user_mat);
    std::map<size_t, double> item_avg_score = get_avg_score_by_row(item_mat);

    auto similar_score_map =
            get_top_k_similar_mat(user_mat, 500, user_avg_score);

    std::vector<Item> result;

    for (size_t test_user_id: test_user_mat.row_indexes()) {
        double bias_user = user_avg_score[test_user_id] - global_avg_score;
        for (const Item &item: test_user_mat.get_row(test_user_id)) {
            const size_t &item_id = item.col;
            double bias_item = item_avg_score[item_id] - global_avg_score;
            double score_base = global_avg_score + bias_user + bias_item;

            double numerator = 0;
            double denominator = 0;
            for (const auto &[similar_user, similarity]:
                    similar_score_map[test_user_id]) {

                // if the similar user has rated the item
                double similar_user_score = user_mat.get(similar_user, item_id);
                if (similar_user_score < 0) {
                    continue;
                }

                double bias_similar_user =
                        user_avg_score[similar_user] - global_avg_score;

                double similar_score_base =
                        global_avg_score + bias_similar_user + bias_item;

                numerator += similarity * (
                        similar_user_score - similar_score_base);
                denominator += std::abs(similarity);
            }

            double score = score_base;
            if (denominator >=
                std::numeric_limits<double>::epsilon()) {
                score += numerator / denominator;
            }
            score = std::clamp(score, 0.0, 100.0);
            result.emplace_back(test_user_id, item_id, score);
        }
    }
    return SparseMatrix<double>(result);
}

double RMSE(const SparseMatrix<double> &mat1,
            const SparseMatrix<double> &mat2) {


    std::span<const Item> mat1_rows = mat1.get_all();
    std::span<const Item> mat2_rows = mat2.get_all();

    if (mat1_rows.size() != mat2_rows.size()) {
        throw std::runtime_error("RMSE size not equal");
    }

    double sum = 0;
    size_t count = mat1_rows.size();

    for (size_t i = 0; i < count; ++i) {
        const Item &predict_item = mat1_rows[i];
        const Item &real_item = mat2_rows[i];
        if (predict_item.row != real_item.row ||
            predict_item.col != real_item.col) {
            throw std::runtime_error("RMSE row or col not equal");
        }
        sum += square(predict_item.val - real_item.val);
    }

    return std::sqrt(sum / count);
}
