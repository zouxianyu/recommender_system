#include <iostream>
#include <iomanip>
#include <cxxopts.hpp>
#include "core.hpp"

void doing(const std::string &str) {
    std::cout << std::setw(60) << std::left << str << " ... " << std::flush;
}

void done() {
    std::cout << "done" << std::endl;
}

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options("recommender_system", "recommender system");
        options.add_options()
                ("e,evaluate", "evaluate RMSE",
                 cxxopts::value<bool>()->default_value("false"))
                ("T,train", "train dataset",
                 cxxopts::value<std::string>()->default_value("train.txt"))
                ("t,test", "test dataset",
                 cxxopts::value<std::string>()->default_value("test.txt"))
                ("a,attribute", "item attribute",
                 cxxopts::value<std::string>()->default_value(
                         "itemAttribute.txt"))
                ("r,result", "result",
                 cxxopts::value<std::string>()->default_value("result.txt"))
                ("k,kusers", "k similar users",
                 cxxopts::value<int>()->default_value("5000"))
                ("use-attribute", "use item attribute",
                 cxxopts::value<bool>()->default_value("false"))
                ("use-weight", "use item attribute weight",
                 cxxopts::value<bool>()->default_value("false"))
                ("h,help", "help");
        auto cmd = options.parse(argc, argv);

        if (cmd.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        bool evaluate = cmd["evaluate"].as<bool>();
        std::string train_filename = cmd["train"].as<std::string>();
        std::string test_filename = cmd["test"].as<std::string>();
        std::string attr_filename = cmd["attribute"].as<std::string>();
        std::string result_filename = cmd["result"].as<std::string>();
        int k = cmd["kusers"].as<int>();
        int flags = 0;
        if (cmd["use-attribute"].as<bool>()) {
            flags |= FEAT_USE_ATTR;
        }
        if (cmd["use-weight"].as<bool>()) {
            flags |= FEAT_USE_WEIGHT;
        }

        // sanity check
        if ((flags & FEAT_USE_WEIGHT) && !(flags & FEAT_USE_ATTR)) {
            throw std::runtime_error("use-weight requires use-attribute");
        }

        // output parameters
        std::cout << "parameters:" << std::endl
                  << "evaluate      = " << std::boolalpha
                  << evaluate << std::endl
                  << "train         = " << train_filename << std::endl
                  << "test          = " << test_filename << std::endl
                  << "attribute     = " << attr_filename << std::endl
                  << "result        = " << result_filename << std::endl
                  << "kusers        = " << k << std::endl
                  << "use-attribute = " << std::boolalpha
                  << !!(flags & FEAT_USE_ATTR) << std::endl
                  << "use-weight    = " << std::boolalpha
                  << !!(flags & FEAT_USE_WEIGHT) << std::endl;

        doing("reading train dataset");
        auto all_dataset = read_train_dataset(train_filename);
        done();

        std::cout << "statistics:" << std::endl
                  << "users   = " << all_dataset.row_indexes().size()
                  << std::endl
                  << "items   = "
                  << all_dataset.transpose().row_indexes().size()
                  << std::endl
                  << "ratings = " << all_dataset.get_all().size()
                  << std::endl;

        doing("reading item attributes");
        auto item_attribute = read_item_attribute(attr_filename);
        done();

        if (evaluate) {
            doing("making train and test dataset");
            auto [train_dataset, test_dataset] =
                    make_train_test(all_dataset, 3);
            done();

            auto result = predict(train_dataset, test_dataset, item_attribute,
                                  k, flags);

            std::cout << "RMSE = " << RMSE(result, test_dataset) << std::endl;

            doing("writing result");
            write_dataset(result_filename, result);
            done();
        } else {
            doing("reading test dataset");
            auto test_dataset = read_test_dataset(test_filename);
            done();

            auto result = predict(all_dataset, test_dataset, item_attribute,
                                  k, flags);

            doing("writing result");
            write_dataset_in_order(test_filename, result_filename, result);
            done();
        }
    } catch (const std::exception &e) {
        std::cout << "error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "unknown error" << std::endl;
        return 1;
    }
    return 0;
}
