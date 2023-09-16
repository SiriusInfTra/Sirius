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
  }

  std::map<std::string, double> poisson;
  std::map<std::string, size_t> interval_ms;
  std::map<std::string, std::vector<double>> change_time_points;
  std::map<std::string, std::vector<size_t>> dynamic_concurrencys;
  std::map<std::string, std::vector<double>> dynamic_poissons;
};

int main(int argc, char** argv) {
  App app;
  CLI11_PARSE(app.app, argc, argv);

  std::string target = "localhost:" + app.port;
  colserve::workload::Workload workload(
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials()),
      std::chrono::seconds(app.duration)
  );
  CHECK(workload.Hello());

  // construct workload
  if (app.enable_infer) {
    if (app.infer_models.count("mnist")) {
      if (!app.interval_ms.count("mnist"))
        workload.InferMnist(app.concurrency, nullptr, app.show_result);
      else
        workload.InferMnist(
            app.concurrency, 
            [&](size_t) {return colserve::workload::double_ms_t(app.interval_ms["mnist"]);},
            app.show_result);
    }
    if (app.infer_models.count("mnist-p")) {
      if (app.poisson.count("mnist"))
        workload.InferMnistPoisson(app.concurrency, app.poisson["mnist"], app.show_result);
      else
        LOG(WARNING) << "mnist-p miss poisson dist parameter";
    }

    if (app.infer_models.count("resnet")) {
      if (!app.interval_ms.count("resnet")) {
        workload.InferResnet("resnet152", app.concurrency, nullptr, app.show_result);
      } else {
        workload.InferResnet(
            "resnet152",
            app.concurrency, 
            [&](size_t) {return colserve::workload::double_ms_t(app.interval_ms["resnet"]);},
            app.show_result);
      }
    }
    if (app.infer_models.count("resnet-p")) {
      if (app.poisson.count("resnet"))
        workload.InferResnetPoisson(app.concurrency, app.poisson["resnet"], app.show_result);
      else
        LOG(FATAL) << "resnet-p miss poisson dist parameter";
    }
    if (app.infer_models.count("resnet-d")) {
      if (app.dynamic_concurrencys.count("resnet") && app.change_time_points.count("resnet")) {
        auto& change_times = app.change_time_points["resnet"];
        auto& concurrencys = app.dynamic_concurrencys["resnet"];
        // for (auto t : change_times) std::cout << t << " ";
        // for (auto c : concurrencys) std::cout << c << " "; std::cout << std::endl;
        CHECK_EQ(concurrencys.size(), change_times.size() + 1);
        workload.InferResnetDynamic(change_times, concurrencys, app.show_result);
      } else {
        LOG(FATAL) << "resnet-d miss dynamic config parameter";
      }
    }
    if (app.infer_models.count("resnet-dp")) {
      CHECK(app.dynamic_poissons.count("resnet") && app.change_time_points.count("resnet"));
      auto &change_times = app.change_time_points["resnet"];
      auto &lambda = app.dynamic_poissons["resnet"];
      CHECK_EQ(lambda.size(), change_times.size() + 1);
      workload.InferResnetDynamicPoisson(app.concurrency, change_times, lambda, app.show_result);
    }
      // workload.InferResnet(app.concurrency, nullptr);
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