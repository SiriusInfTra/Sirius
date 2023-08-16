#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>
#include "model_infer_store.h"
#include <glog/logging.h>

#include "model_train_store.h"
#include "controller.h"
#include "config.h"


namespace colserve {

std::unique_ptr<ModelTrainStore> ModelTrainStore::model_train_store_;

void ModelTrainStore::Init(const std::filesystem::path &train_store_path) {
  model_train_store_ = std::make_unique<ModelTrainStore>();

  for (auto train_script : std::filesystem::directory_iterator(train_store_path)) {
    if (train_script.is_regular_file()) {
      model_train_store_->train_handles_[train_script.path().stem().string()] = train_script.path();
      LOG(INFO) << "ModelTrainStore: Add " << train_script.path().stem().string();
    }
  }

  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, 2);
  model_train_store_->thread_.reset(new std::thread([&]() {
    pthread_barrier_wait(&barrier);
    LOG(INFO) << "ModelTrainStore thread start";
    while (true) {
      model_train_store_->Train();
    }
  }));
  pthread_barrier_wait(&barrier);

  LOG(INFO) << "ModelTrainStore initialized"; 
}

bool ModelTrainStore::Shutdown() {
  if (model_train_store_->train_pid_ != -1) {
    kill(model_train_store_->train_pid_, SIGKILL);
    waitpid(model_train_store_->train_pid_, NULL, 0);
  }
  return true;
}

bool ModelTrainStore::AddJob(network::TrainHandler::TrainData* data) {
  job_queue_.Put(std::make_shared<TrainJob>(data));
  return true;
}

bool ModelTrainStore::Train() {
  std::shared_ptr<Job> job = nullptr;
  while (job == nullptr) {
    job = job_queue_.Get();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  auto data = job->GetTrainData();
  auto model = data->GetModelName();
  auto train_script = train_handles_[model];

  std::vector<std::string> args_str;
  args_str.push_back("python");
  args_str.push_back(train_script.string());
  auto args = data->GetTrainArgs();
  for (size_t i = 0, j = 0; i < args.size(); ) {
    for (j = i; j < args.size() && args[j] != '='; j++);
    args_str.push_back("--" + args.substr(i, j - i));
    for (i = j + 1, j = i; j < args.size() && args[j] != ' ' && args[j] != ','; j++); 
    args_str.push_back(args.substr(i, j - i));
    for (i = j + 1; i < args.size() && args[i] == ' '; i++);
  }
  if (Config::serve_mode == ServeMode::kTaskSwitchL1) {
    args_str.push_back("--mode");
    args_str.push_back("task-switch");
  }

  std::stringstream ss;
  char* argv[args_str.size() + 1];
  for (size_t i = 0; i < args_str.size(); i++) {
    argv[i] = args_str[i].data();
    ss << args_str[i] << " ";
  }
  argv[args_str.size()] = 0;
  
  int from_child_pipe[2], to_child_pipe[2];
  pipe(from_child_pipe);
  pipe(to_child_pipe);

  if (Config::serve_mode == ServeMode::kTaskSwitchL1) {
    Controller::Get()->WaitInferIdle();
  }

  auto pid = fork();
  train_pid_ = pid;
  CHECK_GE(pid, 0) << "ModelTrainStore: fork failed";

  if (pid == 0) {
    close(to_child_pipe[1]);
    dup2(to_child_pipe[0], STDIN_FILENO);
    close(from_child_pipe[0]);
    dup2(from_child_pipe[1], STDOUT_FILENO);

    auto err = execvp("python", argv);
    perror("execvp");
    CHECK_GE(err, 0) << "ModelTrainStore: spawn train worker fail, errno " << err;
  } else {
    close(to_child_pipe[0]);
    close(from_child_pipe[1]);
    // train_running_ = true;
    // Controller::Get()->TrainStart();
    LOG(INFO) << "ModelTrainStore: " << "Train " << job << " ( "
              << ss.str() << "), pid " << pid;
  }

  std::array<char, 1024> buf;
  auto fp = fdopen(from_child_pipe[0], "r");

  while (fgets(buf.data(), buf.size(), fp)) {
    LOG(INFO) << "[PyTorch backend] Train: " << buf.data();
  }

  fclose(fp);
  close(to_child_pipe[1]);
  close(from_child_pipe[0]);
  waitpid(pid, NULL, 0);

  train_pid_ = -1;
  LOG(INFO) << "Train: " << job << " finished";
  Controller::Get()->TrainEnd(); // double check train end

  data->SetResult("train ok");
  data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
  return true;
}

} // namespace colserve