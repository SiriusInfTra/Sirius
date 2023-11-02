#include <random>
#include <regex>

#include "colserve.grpc.pb.h"
#include "workload/workload.h"

struct App : public colserve::workload::AppBase {
  App() : AppBase() {
    app.add_option("--poisson", poisson,
         "poisson distribution of infer models");
    app.add_option("--interval", interval_ms, 
        "interval (ms) of infer models");
    app.add_option("--change-time-point", change_time_points, 
        "dynamic concurrency change time point ");
    app.add_option("--dynamic-concurrency", dynamic_concurrencys,
        "dynamic concurrencys");
    app.add_option("--dynamic-poisson", dynamic_poissons,
        "dynamic poisson distribution of infer models");

    auto benchmark_group = app.add_option_group("benchmark");
    benchmark_group->add_flag("--benchmark", benchmark.enable, "enable benchmark");
    benchmark_group->add_option("--random-dynamic-poisson", 
                                benchmark.random_dynamic_possion, "random dynamic poisson");
    // app.add_option("--benchmark-poisson", , "micro benchmark");
  }

  std::map<std::string, double> poisson;
  std::map<std::string, size_t> interval_ms;
  std::map<std::string, std::vector<double>> change_time_points;
  std::map<std::string, std::vector<size_t>> dynamic_concurrencys;
  std::map<std::string, std::vector<double>> dynamic_poissons;
  
  struct {
    bool enable{false};
    double random_dynamic_possion;
  } benchmark;
};

std::vector<std::pair<std::string, std::string>> ParseModelName(const std::string &model) {
  std::regex r{"([a-zA-Z0-9]+)(-([a-zA-Z0-9]+))?(\\[([0-9]+)\\])?"};
  std::smatch match;
  CHECK(std::regex_match(model, match, r));
  std::cout.flush();
  CHECK(match.size() == 6);
  if (match[5].str().empty()) {
    return {{match[1].str(), match[3].str()}};
  } else {
    std::vector<std::pair<std::string, std::string>> ret{{match[1].str(), match[3].str()}};
    for (int i = 1; i < std::stoi(match[5].str()); i++) {
      ret.emplace_back(match[1].str()+"-"+std::to_string(i), match[3].str());
    }
    return ret;
  }
}

int main(int argc, char** argv) {
  App app;
  CLI11_PARSE(app.app, argc, argv);

  std::string target = "localhost:" + app.port;
  colserve::workload::Workload workload(
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials()),
      std::chrono::seconds(app.duration)
  );
  CHECK(workload.Hello());

  std::vector<std::pair<std::string, std::string>> infer_models;
  for (auto &m : app.infer_models) {
    for (auto &model : ParseModelName(m)) {
      infer_models.push_back(model);
    }
  }


  if (app.seed == static_cast<uint64_t>(-1)) {
    std::random_device rd;
    app.seed = rd();
  }
  LOG(INFO) << "Workload random seed " << app.seed;


  if (app.enable_infer) {
    if (app.benchmark.enable) {
      std::vector<std::string> models;
      for (auto &m : infer_models) { 
        models.push_back(m.first); 
      }
      std::mt19937 gen;
      gen.seed(app.seed);
      std::uniform_real_distribution<> lambda_dist(0, app.benchmark.random_dynamic_possion);
      std::uniform_int_distribution<> req_model_num_dist(0, infer_models.size());
      std::vector<double> total_lambda{0};
      std::map<std::string, std::vector<double>> model_lambdas;
      CHECK(app.change_time_points.count("benchmark"));
      auto change_time_points = app.change_time_points["benchmark"];
      for (size_t i = 0; i < change_time_points.size(); i++) {
        total_lambda.push_back(lambda_dist(gen));
      }
      
      for (size_t i = 0; i <= change_time_points.size(); i++) {
        int req_model_num = req_model_num_dist(gen);
        std::vector<double> cur_lambda;
        std::uniform_real_distribution<> dist(0, 1);
        for (int j = 0; j < req_model_num; j++) {
          cur_lambda.push_back(dist(gen));
        }
        auto acc = std::accumulate(cur_lambda.begin(), cur_lambda.end(), 0.0);
        std::for_each(cur_lambda.begin(), cur_lambda.end(), [s=acc,i, &total_lambda](double &x) {
          if (s == 0) x = 0; else x /= s;
          x *= total_lambda[i];
        });
        std::shuffle(models.begin(), models.end(), gen);
        for (size_t j = 0; j < models.size(); j++) {
          if (j < req_model_num) {
            model_lambdas[models[j]].push_back(cur_lambda[j]);
          } else {
            model_lambdas[models[j]].push_back(0);
          }
        }
      }
      std::cerr << "Benchmarh Random Dynamic Lambdas: ";
      for (size_t i = 0; i < total_lambda.size(); i++) {
        if (i == 0) std::cerr << std::fixed << std::setprecision(1) << total_lambda[i] << "\t";
        else std::cerr << total_lambda[i] << "(" << change_time_points[i-1] << ")\t";
      }
      std::cerr << std::endl;
      for (size_t i = 0; i < models.size(); i++) {
        std::cerr << "  " << std::setw(15) << models[i] << " lambdas: ";
        for (auto l : model_lambdas[models[i]]) {
          std::cerr << std::fixed << std::setprecision(1) << std::setw(4) << l << " ";
        }
        std::cerr << std::endl;
        workload.InferDynamicPoisson(models[i], app.concurrency, change_time_points, model_lambdas[models[i]]);
      }

    } else {
      for (auto &m : infer_models) {
        auto [model, type] = m;
        if (type.empty()) {
          if (app.interval_ms.count(model)) 
            workload.Infer(model, app.concurrency,
                           [&](size_t){return colserve::workload::double_ms_t(app.interval_ms[model]);},
                           app.show_result);
          else
            workload.Infer(model, app.concurrency, nullptr, app.show_result);
        } else if (type == "p") {
          CHECK(app.poisson.count(model));
          workload.InferPoisson(model, app.concurrency, app.poisson[model], app.show_result);
        } else if (type == "d") {
          CHECK(app.change_time_points.count(model));
          CHECK(app.dynamic_concurrencys.count(model));
          workload.InferDynamic(model, app.change_time_points[model], app.dynamic_concurrencys[model], app.show_result);
        } else if (type == "dp") {
          CHECK(app.change_time_points.count(model));
          CHECK(app.dynamic_poissons.count(model));
          workload.InferDynamicPoisson(model, app.concurrency, app.change_time_points[model], app.dynamic_poissons[model], app.show_result);
        } else {
          LOG(FATAL) << "unknown workload " << model << " type " << type ;
        }
      }
      // if (app.infer_models.count("mnist")) {
      //   if (!app.interval_ms.count("mnist"))
      //     workload.InferMnist(app.concurrency, nullptr, app.show_result);
      //   else
      //     workload.InferMnist(
      //         app.concurrency, 
      //         [&](size_t) {return colserve::workload::double_ms_t(app.interval_ms["mnist"]);},
      //         app.show_result);
      // }
      // if (app.infer_models.count("mnist-p")) {
      //   if (app.poisson.count("mnist"))
      //     workload.InferMnistPoisson(app.concurrency, app.poisson["mnist"], app.show_result);
      //   else
      //     LOG(WARNING) << "mnist-p miss poisson dist parameter";
      // }

      // if (app.infer_models.count("resnet")) {
      //   if (!app.interval_ms.count("resnet")) {
      //     workload.InferResnet("resnet152", app.concurrency, nullptr, app.show_result);
      //   } else {
      //     workload.InferResnet(
      //         "resnet152",
      //         app.concurrency, 
      //         [&](size_t) {return colserve::workload::double_ms_t(app.interval_ms["resnet"]);},
      //         app.show_result);
      //   }
      // }
      // if (app.infer_models.count("resnet-p")) {
      //   if (app.poisson.count("resnet"))
      //     workload.InferResnetPoisson(app.concurrency, app.poisson["resnet"], app.show_result);
      //   else
      //     LOG(FATAL) << "resnet-p miss poisson dist parameter";
      // }
      // if (app.infer_models.count("resnet-d")) {
      //   if (app.dynamic_concurrencys.count("resnet") && app.change_time_points.count("resnet")) {
      //     auto& change_times = app.change_time_points["resnet"];
      //     auto& concurrencys = app.dynamic_concurrencys["resnet"];
      //     CHECK_EQ(concurrencys.size(), change_times.size() + 1);
      //     workload.InferResnetDynamic(change_times, concurrencys, app.show_result);
      //   } else {
      //     LOG(FATAL) << "resnet-d miss dynamic config parameter";
      //   }
      // }
      // if (app.infer_models.count("resnet-dp")) {
      //   CHECK(app.dynamic_poissons.count("resnet") && app.change_time_points.count("resnet"));
      //   auto &change_times = app.change_time_points["resnet"];
      //   auto &lambda = app.dynamic_poissons["resnet"];
      //   CHECK_EQ(lambda.size(), change_times.size() + 1);
      //   workload.InferResnetDynamicPoisson(app.concurrency, change_times, lambda, app.show_result);
      // }
      // workload.InferResnet(app.concurrency, nullptr);
    }
  }
  
  if (app.enable_train) {
    if (app.train_models.count("resnet"))
      workload.TrainResnet(app.num_epoch, app.batch_size);
  }

  workload.Run();

  LOG(INFO) << "report result ...";
  if (app.log.empty()) {
    workload.Report(app.verbose);
  } else {
    std::fstream ofs{app.log, std::ios::out};
    CHECK(ofs.good());
    workload.Report(app.verbose, ofs);
    ofs.close();
  }
  return 0;
}