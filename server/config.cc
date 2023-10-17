#include "config.h"

namespace colserve {

ServeMode Config::serve_mode = ServeMode::kNormal;

std::atomic<bool> Config::running = true;

bool Config::use_shared_tensor = true;

}