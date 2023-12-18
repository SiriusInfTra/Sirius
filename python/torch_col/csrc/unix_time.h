#include <chrono>

namespace torch_col {
inline auto get_unix_timestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

}
