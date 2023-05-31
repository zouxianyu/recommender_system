#include <iostream>
#include "core.hpp"

int main() {
    std::cout << "reading train dataset" << std::endl;
    auto all_dataset = read_train_dataset(R"(D:\source\CLionProjects\big_data\recommender_system\data\train.txt)");
    std::cout << "read train dataset finished" << std::endl;

//    std::cout << "reading test dataset" << std::endl;
//    auto test_dataset = read_test_dataset(R"(D:\source\CLionProjects\big_data\recommender_system\data\test.txt)");
//    std::cout << "read test dataset finished" << std::endl;

    std::cout << "reading item attributes" << std::endl;
    auto item_attribute = read_item_attribute(R"(D:\source\CLionProjects\big_data\recommender_system\data\itemAttribute.txt)");
    std::cout << "read item attributes finished" << std::endl;

    std::cout << "splitting train dataset" << std::endl;
    auto [train_dataset, test_dataset] = make_train_test(all_dataset, 3);

//    auto row = test_dataset.get_row(19834);
//    std::cout << "user[<last>].size = " << row.size() << std::endl;
//    for (auto item: row) {
//        std::cout << item.row << " " << item.col << " " << item.val << std::endl;
//    }

    auto result = solve(train_dataset, test_dataset);

    std::cout << "RMSE = " << RMSE(result, test_dataset) << std::endl;

    std::cout << "writing result" << std::endl;
    write_dataset(R"(D:\source\CLionProjects\big_data\recommender_system\data\result.txt)", result);
    std::cout << "write result finished" << std::endl;

    return 0;
}
