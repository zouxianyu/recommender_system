#include <fstream>
#include <stdexcept>
#include <map>
#include <algorithm>
#include <iostream>
#include <vector>
#include <array>
#include <indicators/progress_bar.hpp>
#include "core.hpp"

using namespace indicators;

using FpItem = SparseMatrix<double>::Item;
using IntItem = SparseMatrix<int>::Item;

/**
 * read dataset from file (train or test)
 * @param filename file name of the dataset
 * @param has_score whether the dataset has score
 * @return the dataset stored in SparseMatrix
 */
SparseMatrix<double> read_dataset(const std::string &filename, bool has_score) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    std::vector<FpItem> items;
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

/**
 * read train dataset from file (wrapper)
 * @param filename file name of the dataset
 * @return the dataset stored in SparseMatrix
 */
SparseMatrix<double> read_train_dataset(const std::string &filename) {
    return read_dataset(filename, true);
}

/**
 * read test dataset from file (wrapper)
 * @param filename file name of the dataset
 * @return the dataset stored in SparseMatrix
 */
SparseMatrix<double> read_test_dataset(const std::string &filename) {
    return read_dataset(filename, false);
}

/**
 * read item attribute from file
 * @param filename file name of the item attribute
 * @return item attribute stored in SparseMatrix, '1' for attribute exists
 */
SparseMatrix<int> read_item_attribute(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }

    std::vector<IntItem> items;
    size_t item_id;
    std::string line;
    while (!file.eof()) {
        char split;
        file >> item_id >> split;
        std::getline(file, line);
        if (line.empty()) {
            continue;
        }
        auto pos = line.find_first_of('|');
        if (pos == std::string::npos) {
            throw std::runtime_error("Item attribute file format error");
        }
        std::string attr1_str = line.substr(0, pos);
        std::string attr2_str = line.substr(pos + 1);

        if (attr1_str != "None") {
            items.emplace_back(item_id, std::stoi(attr1_str), 1);
        }
        if (attr2_str != "None") {
            items.emplace_back(item_id, std::stoi(attr2_str), 1);
        }
    }
    return SparseMatrix<int>(items);
}

/**
 * write result to file
 * @param filename file name of the result
 * @param mat result stored in SparseMatrix
 */
void write_dataset(const std::string &filename,
                   const SparseMatrix<double> &mat) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }

    for (size_t row_id: mat.row_indexes()) {
        std::span<const FpItem> row = mat.get_row(row_id);
        file << row_id << "|" << row.size() << std::endl;
        for (const auto &item: row) {
            file << item.col << "  " << item.val << std::endl;
        }
    }
}

/**
 * split dataset into train and test
 * @param mat the whole dataset
 * @param test_count count of test items for each user
 * @return train and test dataset
 */
std::pair<SparseMatrix<double>, SparseMatrix<double>> make_train_test(
        const SparseMatrix<double> &mat, size_t test_count) {
    std::vector<FpItem> train_items;
    std::vector<FpItem> test_items;

    // the answer to life, the universe and everything
    srand(42);
    size_t seed = rand();

    for (size_t row_id: mat.row_indexes()) {
        std::span<const FpItem> row = mat.get_row(row_id);
        if (row.size() <= test_count) {
            continue;
        }

        for (size_t i = 0; i < row.size(); ++i) {
            size_t next_i = i + row.size();
            size_t base = seed % row.size();

            if ((base <= i && i < base + test_count) ||
                (base <= next_i && next_i < base + test_count)) {
                test_items.emplace_back(row[i]);
            } else {
                train_items.emplace_back(row[i]);
            }
        }

    }
    return {SparseMatrix<double>(train_items),
            SparseMatrix<double>(test_items)};
}

/**
 * get average score for each row (user / item)
 * @param mat dataset
 * @return average score for each row (represented by map)
 */
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

/**
 * get average score for the whole dataset
 * @param mat dataset
 * @return average score
 */
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

/**
 * calculate pearson correlation between two rows (user / item)
 * @param mat dataset
 * @param x the first row
 * @param y the second row
 * @param avg_score cached average score for each row
 * @return pearson correlation between two rows
 */
double pearson(const SparseMatrix<double> &mat, size_t x, size_t y,
               const std::map<size_t, double> &avg_score) {
    std::span<const FpItem> row_x = mat.get_row(x);
    std::span<const FpItem> row_y = mat.get_row(y);
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

/**
 * comparator for the top-k top-k items with highest score
 * (for min heap)
 * @param a
 * @param b
 * @return compare result
 */
bool heap_compare(const std::pair<size_t, double> &a,
                  const std::pair<size_t, double> &b) {
    return a.second > b.second;
}

/**
 * update top-k items with highest score
 * @param top_k top-k items with highest score
 * @param k k value
 * @param id new item id
 * @param score item's score
 */
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

/**
 * make similarity matrix
 * @param mat dataset
 * @param k k value
 * @param avg_score cached average score for each row
 * @return similarity matrix (represented by map)
 */
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
            option::PrefixText{"Train  "},
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

/**
 * get similar items of a given item
 * @param item_id item id to find similar items
 * @param item_attr item attribute matrix (item -> attribute)
 * @param item_attr_rev reverse item attribute matrix (attribute -> item)
 * @return similar items split by attribute
 */
std::array<std::span<const IntItem>, 2> get_similar_items(
        size_t item_id,
        const SparseMatrix<int> &item_attr,
        const SparseMatrix<int> &item_attr_rev
) {
    std::array<std::span<const IntItem>, 2> result;
    std::span<const IntItem> attrs = item_attr.get_row(item_id);
    for (size_t i = 0; i < attrs.size(); ++i) {
        // find which item has the same attribute id
        const size_t &attr_id = attrs[i].col;
        std::span<const IntItem> entries = item_attr_rev.get_row(attr_id);
        result[i] = entries;
    }
    return result;
}

/**
 * predict score of a given item
 * @param user_id user id to predict_impl
 * @param item_id item id to predict_impl
 * @param user_mat user matrix (user -> item score)
 * @param global_avg_score cached global average score
 * @param user_avg_score cached average score for each user
 * @param item_avg_score cached average score for each item
 * @param similar_score_map cached similar score map
 * @param item_attr item attribute matrix (item -> attribute)
 * @param item_attr_rev reverse item attribute matrix (attribute -> item)
 * @param consider_similar_items whether it is the first try,
 *                  determine whether to calculate similar items
 * @return predicted score
 */
double predict_impl(
        size_t user_id,
        size_t item_id,
        const SparseMatrix<double> &user_mat,
        double global_avg_score,
        std::map<size_t, double> &user_avg_score,
        std::map<size_t, double> &item_avg_score,
        std::map<size_t, std::vector<std::pair<size_t, double>>> &similar_score_map,
        const SparseMatrix<int> &item_attr,
        const SparseMatrix<int> &item_attr_rev,
        bool consider_similar_items
) {
    double bias_user = user_avg_score[user_id] - global_avg_score;
    double bias_item = item_avg_score[item_id] - global_avg_score;
    double score_base = global_avg_score + bias_user + bias_item;

    double numerator = 0;
    double denominator = 0;
    size_t count = 0;
    for (const auto &[similar_user, similarity]:
            similar_score_map[user_id]) {

        // if the similar user has rated the item
        double similar_user_score = user_mat.get(similar_user, item_id);
        if (similar_user_score < 0) {
            continue;
        }
        count++;

        double bias_similar_user =
                user_avg_score[similar_user] - global_avg_score;

        double similar_score_base =
                global_avg_score + bias_similar_user + bias_item;

        numerator += similarity * (similar_user_score - similar_score_base);
        denominator += std::abs(similarity);
    }

    double score;

    if (denominator < std::numeric_limits<double>::epsilon() || count <= 1) {
        // similar users not enough

        // if it is the first try, calculate similar items
        // else just abandon this similar item
        if (!consider_similar_items) {
            return -1;
        }

        double similar_item_score_nominator = 0;
        double similar_item_score_denominator = 0;
        for (std::span<const IntItem> items: get_similar_items(
                item_id, item_attr, item_attr_rev)) {

            // except the item itself
            size_t similar_item_count = items.size() - 1;
            if (similar_item_count <= 0) {
                continue;
            }

            double attr_weight = 1.0 / similar_item_count;

            for (const IntItem &entry: items) {
                const size_t &similar_item_id = entry.col;

                // skip the item itself
                if (similar_item_id == item_id) {
                    continue;
                }

                // first try: get similar item score from user matrix directly
                //            which is faster and more accurate
                double similar_item_score = user_mat.get(
                        user_id, similar_item_id);

                // second try: try to predict similar item score
                //             by recursively calling predict()
                if (similar_item_score < 0) {
                    similar_item_score = predict_impl(
                            user_id,
                            similar_item_id,
                            user_mat,
                            global_avg_score,
                            user_avg_score,
                            item_avg_score,
                            similar_score_map,
                            item_attr,
                            item_attr_rev,
                            false
                    );
                }

                // failed: skip the similar item
                if (similar_item_score < 0) {
                    continue;
                }

                // success: add the similar item score with attribute weight
                similar_item_score_nominator +=
                        attr_weight * similar_item_score;
                similar_item_score_denominator += attr_weight;
            }

        }

        if (similar_item_score_denominator >
            std::numeric_limits<double>::epsilon()) {
            // have enough similar items to calculate predict score
            score = similar_item_score_nominator /
                    similar_item_score_denominator;
        } else {
            // no similar items, use user base score
            score = score_base;
        }
    } else {
        score = score_base + numerator / denominator;
    }

    score = std::clamp(score, 0.0, 100.0);
    return score;
}

/**
 * solve the problem
 * @param user_mat train dataset
 * @param test_user_mat test dataset
 * @param item_attr item attribute matrix (item -> attribute)
 * @return predicted score matrix
 */
SparseMatrix<double> predict(const SparseMatrix<double> &user_mat,
                             const SparseMatrix<double> &test_user_mat,
                             const SparseMatrix<int> &item_attr) {

    SparseMatrix<double> item_mat = user_mat.transpose();

    double global_avg_score = get_global_avg_score(user_mat);
    std::map<size_t, double> user_avg_score = get_avg_score_by_row(user_mat);
    std::map<size_t, double> item_avg_score = get_avg_score_by_row(item_mat);

    SparseMatrix<int> item_attr_rev = item_attr.transpose();

    auto similar_score_map =
            get_top_k_similar_mat(user_mat, 5000, user_avg_score);

    // info for progress bar
    const size_t all_count = test_user_mat.get_all().size();
    size_t current_count = 0;
    ProgressBar bar{
            option::PrefixText{"Predict"},
            option::BarWidth{50},
            option::ShowPercentage{true},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
    };

    std::vector<FpItem> result;

    for (size_t test_user_id: test_user_mat.row_indexes()) {
        for (const FpItem &item: test_user_mat.get_row(test_user_id)) {
            const size_t &item_id = item.col;

            double score = predict_impl(
                    test_user_id,
                    item_id,
                    user_mat,
                    global_avg_score,
                    user_avg_score,
                    item_avg_score,
                    similar_score_map,
                    item_attr,
                    item_attr_rev,
                    true
            );

            result.emplace_back(test_user_id, item_id, score);

            // show progress bar
            double progress = static_cast<double>(++current_count) / all_count;
            if (current_count == all_count || current_count % 100 == 0) {
                bar.set_progress(progress * 100);
            }
        }
    }
    return SparseMatrix<double>(result);
}

/**
 * calculate RMSE between two matrix (same size)
 * @param mat1
 * @param mat2
 * @return RMSE
 */
double RMSE(const SparseMatrix<double> &mat1,
            const SparseMatrix<double> &mat2) {

    std::span<const FpItem> mat1_rows = mat1.get_all();
    std::span<const FpItem> mat2_rows = mat2.get_all();

    if (mat1_rows.size() != mat2_rows.size()) {
        throw std::runtime_error("RMSE size not equal");
    }

    double sum = 0;
    size_t count = mat1_rows.size();

    for (size_t i = 0; i < count; ++i) {
        const FpItem &predict_item = mat1_rows[i];
        const FpItem &real_item = mat2_rows[i];
        if (predict_item.row != real_item.row ||
            predict_item.col != real_item.col) {
            throw std::runtime_error("RMSE row or col not equal");
        }
        sum += square(predict_item.val - real_item.val);
    }

    return std::sqrt(sum / count);
}
