#include <iostream>
#include <iomanip>
#include "core.hpp"

inline void doing(const std::string &str) {
    std::cout << std::setw(60) << std::left << str << " ... " << std::flush;
}

inline void done() {
    std::cout << "done" << std::endl;
}

int main() {
    doing("reading train dataset");
    auto all_dataset = read_train_dataset(R"(D:\source\CLionProjects\big_data\recommender_system\data\train.txt)");
    done();

//    std::cout << "reading test dataset" << std::endl;
//    auto test_dataset = read_test_dataset(R"(D:\source\CLionProjects\big_data\recommender_system\data\test.txt)");
//    std::cout << "read test dataset finished" << std::endl;

    doing("reading item attributes");
    auto item_attribute = read_item_attribute(R"(D:\source\CLionProjects\big_data\recommender_system\data\itemAttribute.txt)");
    done();

    doing("making train and test dataset");
    auto [train_dataset, test_dataset] = make_train_test(all_dataset, 3);
    done();

    auto result = solve(train_dataset, test_dataset, item_attribute);

    std::cout << "RMSE = " << RMSE(result, test_dataset) << std::endl;

    doing("writing result");
    write_dataset(R"(D:\source\CLionProjects\big_data\recommender_system\data\result.txt)", result);
    done();

    return 0;
}
