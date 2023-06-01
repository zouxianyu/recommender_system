#ifndef RECOMMENDER_SYSTEM_SPARSE_MATRIX_HPP
#define RECOMMENDER_SYSTEM_SPARSE_MATRIX_HPP

#include <tuple>
#include <vector>
#include <algorithm>
#include <span>
#include <set>

/**
 * sparse matrix for storing data
 * @tparam T
 */
template<typename T>
class SparseMatrix {
public:
    struct Item {
        size_t row;
        size_t col;
        T val;

        bool operator<(const Item &other) const {
            return std::tie(row, col) <
                   std::tie(other.row, other.col);
        }
    };

    /**
     * constructor
     * construct sparse matrix from unordered items
     * @param unordered_items
     */
    explicit SparseMatrix(std::vector<Item> unordered_items) {
        for (const auto &item: unordered_items) {
            items.emplace_back(item);
            rows.emplace(item.row);
        }
        std::sort(items.begin(), items.end());
    }

    /**
     * transpose matrix
     * @return transposed matrix
     */
    SparseMatrix transpose() const {
        std::vector<Item> transposed_items;
        for (const auto &item: items) {
            transposed_items.emplace_back(item.col, item.row, item.val);
        }
        return SparseMatrix(transposed_items);
    }

    /**
     * get item by row and col
     * @param row
     * @param col
     * @return item
     */
    T get(size_t row, size_t col) const {
        // binary search by lower_bound and upper_bound
        auto lower = std::lower_bound(items.begin(), items.end(),
                                      Item{row, col, T{}});

        auto upper = std::upper_bound(items.begin(), items.end(),
                                      Item{row, col, T{}});
        if (lower == upper) {
            return -1;
        } else {
            return lower->val;
        }
    }

    /**
     * get row by row index
     * @param row
     * @return view of the row
     */
    std::span<const Item> get_row(size_t row) const {
        // binary search by lower_bound and upper_bound
        auto lower = std::lower_bound(
                items.begin(), items.end(),
                Item{row, std::numeric_limits<size_t>::min(), T{}});
        auto upper = std::upper_bound(
                items.begin(), items.end(),
                Item{row, std::numeric_limits<size_t>::max(), T{}});

        return {lower, upper};
    }

    /**
     * get all items
     * @return view of all items
     */
    std::span<const Item> get_all() const {
        return {items.begin(), items.end()};
    }

    /**
     * get all row indexes
     * @return view of all row indexes
     */
    const std::set<size_t> &row_indexes() const {
        return rows;
    }

private:
    std::vector<Item> items;
    std::set<size_t> rows;
};

#endif //RECOMMENDER_SYSTEM_SPARSE_MATRIX_HPP
