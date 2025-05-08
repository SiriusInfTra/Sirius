#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <cstdint>
#include <functional>

namespace xsched::utils
{

class TcpClient
{
public:
    TcpClient() = default;
    ~TcpClient();

    int Connect(const char *ip, uint16_t port);
    int Connect(const std::string &ip, uint16_t port);
    void Close();
    
    size_t Recv(std::vector<char> &data);
    size_t Send(const void *data, size_t size);

private:
    int conn_fd_ = -1;
};

// void Handler(const void *data, size_t size, std::vector<char> &response);
typedef std::function<void (const void *, size_t,
                            std::vector<char> &)> TcpHandler;

class TcpServer
{
public:
    TcpServer() = default;
    ~TcpServer();

    void Run(uint16_t port, TcpHandler handler);
    void Stop();

private:
    void Worker();

    TcpHandler handler_;
    uint16_t port_ = 0;
    int listen_fd_ = -1;
    std::atomic_bool running_ = { false };
    std::unique_ptr<std::thread> worker_thread_ = nullptr;
};

} // namespace xsched::utils
