#include <glog/logging.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>

#include "model_train_store.h"


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

  std::stringstream ss;
  char* argv[args_str.size() + 1];
  for (size_t i = 0; i < args_str.size(); i++) {
    argv[i] = args_str[i].data();
    ss << args_str[i] << " ";
  }
  argv[args_str.size()] = 0;
  
  int pipe_fd[2];
  pipe(pipe_fd);

  auto pid = fork();
  CHECK_GE(pid, 0) << "ModelTrainStore: fork failed";

  if (pid == 0) {
    close(pipe_fd[0]);
    dup2(pipe_fd[1], STDOUT_FILENO);

    auto err = execvp("python", argv);
    perror("execvp");
    CHECK_GE(err, 0) << "ModelTrainStore: spawn train worker fail, errno " << err;
  } else {
    close(pipe_fd[1]);
    LOG(INFO) << "ModelTrainStore: " << "Train " << job << " ( "
              << ss.str() << "), pid " << pid;
  }

  std::array<char, 128> buf;
  auto fp = fdopen(pipe_fd[0], "r");
  while (fgets(buf.data(), buf.size(), fp)) {
    LOG(INFO) << "[PyTorch backend] Train: " << buf.data();
  }
  fclose(fp);

  waitpid(pid, NULL, 0);
  LOG(INFO) << "Train: " << job << " finished";

  data->SetResult("train ok");
  data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
  return true;
}

} // namespace colserve