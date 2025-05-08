#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>


#include "utils/tcp.h"
#include "utils/xassert.h"

using namespace xsched::utils;

static const size_t kMaxHeaderSize = 128;
static const char kHeader[] = "XSched TCP Socket Header, Length:";
static_assert(sizeof(kHeader) < kMaxHeaderSize, "header size too large");

static size_t RecvHeader(int fd)
{
    char header[kMaxHeaderSize] = { 0 };

    for (size_t i = 0; i < sizeof(kHeader) - 1; ++i) {
        ssize_t n = recv(fd, header + i, 1, MSG_WAITALL);
        if (n == 0) return 0;
        if (n < 0) {
            XWARN("fail to recv header, errno: %d", errno);
            return 0;
        }
        if (header[i] != kHeader[i]) {
            XWARN("invalid header: %s", header);
            return 0;
        }
    }
    
    for (size_t i = 0; i < kMaxHeaderSize; ++i) {
        ssize_t n = recv(fd, header + i, 1, MSG_WAITALL);
        if (n == 0) return 0;
        if (n < 0) {
            XWARN("fail to recv header, errno: %d", errno);
            return 0;
        }
        if (header[i] == '\0') {
            size_t size = std::stoul(header);
            return size;
        }
    }

    XWARN("header too large, length: %s", header);
    return 0;
}

static void SendHeader(int fd, size_t size)
{
    char header[kMaxHeaderSize] = { 0 };
    ssize_t len = sprintf(header, "%s%lu", kHeader, size) + 1;
    for (ssize_t i = 0; i < len;) {
        ssize_t n = send(fd, header + i, len - i, MSG_WAITALL);
        if (n < 0) {
            XWARN("fail to send header, errno: %d", errno);
            return;
        }
        i += n;
    }
}

static size_t RecvData(int fd, void *data, size_t size)
{
    for (size_t i = 0; i < size;) {
        ssize_t n = recv(fd, (char *)data + i, size - i, MSG_WAITALL);
        if (n == 0) return i;
        if (n < 0) {
            XWARN("fail to recv data, errno: %d", errno);
            return i;
        }
        i += n;
    }
    return size;
}

static void SendData(int fd, const void *data, size_t size)
{
    for (size_t i = 0; i < size;) {
        ssize_t n = send(fd, (const char *)data + i, size - i, MSG_WAITALL);
        if (n < 0) {
            XWARN("fail to send data, errno: %d", errno);
            return;
        }
        i += n;
    }
}

TcpClient::~TcpClient()
{
    Close();
}

int TcpClient::Connect(const char *ip, uint16_t port)
{
    conn_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (conn_fd_ < 0) {
        XWARN("fail to create socket, errno: %d", errno);
        return -1;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(ip);
    addr.sin_port = htons(port);

    int ret = connect(conn_fd_, (struct sockaddr *)&addr, sizeof(addr));
    if (ret < 0) {
        XWARN("fail to connect, errno: %d", errno);
        close(conn_fd_);
        conn_fd_ = -1;
        return -1;
    }
    return ret;
}

int TcpClient::Connect(const std::string &ip, uint16_t port)
{
    return Connect(ip.c_str(), port);
}

void TcpClient::Close()
{
    if (conn_fd_ < 0) return;
    close(conn_fd_);
    conn_fd_ = -1;
}

size_t TcpClient::Recv(std::vector<char> &data)
{
    if (conn_fd_ < 0) {
        XWARN("invalid connection");
        return 0;
    }

    size_t size = RecvHeader(conn_fd_);
    if (size == 0) return 0;

    data.resize(size);
    return RecvData(conn_fd_, data.data(), size);
}

size_t TcpClient::Send(const void *data, size_t size)
{
    if (conn_fd_ < 0) {
        XWARN("invalid connection");
        return 0;
    }

    SendHeader(conn_fd_, size);

    if (data == nullptr || size == 0) return 0;
    SendData(conn_fd_, data, size);
    return size;
}

TcpServer::~TcpServer()
{
    Stop();
}

void TcpServer::Run(uint16_t port, TcpHandler handler)
{
    port_ = port;
    handler_ = handler;

    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    XASSERT(listen_fd_ >= 0, "fail to create socket, errno: %d", errno);

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    XASSERT(bind(listen_fd_, (struct sockaddr *)&addr, sizeof(addr)) == 0,
            "fail to bind socket, errno: %d", errno);
    
    running_.store(true);
    worker_thread_ = std::make_unique<std::thread>(&TcpServer::Worker, this);
}

void TcpServer::Stop()
{
    if (!running_.load()) return;

    running_.store(false);
    TcpClient client;
    client.Connect("127.0.0.1", port_);
    client.Send(nullptr, 0);
    client.Close();

    worker_thread_->join();
    worker_thread_ = nullptr;
    close(listen_fd_);
}

void TcpServer::Worker()
{
    XASSERT(listen(listen_fd_, SOMAXCONN) == 0,
            "fail to listen, errno: %d", errno);
    
    while (running_.load()) {
        struct sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);
        int conn_fd = accept(listen_fd_, (struct sockaddr *)&addr, &addr_len);
        XASSERT(conn_fd >= 0, "fail to accept, errno: %d", errno);

        if (!running_.load()) {
            close(conn_fd);
            break;
        }

        size_t size = RecvHeader(conn_fd);
        if (size == 0) {
            close(conn_fd);
            continue;
        }

        void *buf = malloc(size);
        size_t recv_size = RecvData(conn_fd, buf, size);
        if (recv_size != size) {
            XWARN("received: %ld B, expected: %ld B", recv_size, size);
            close(conn_fd);
            continue;
        }

        std::vector<char> response;
        handler_(buf, size, response);
        free(buf);

        if (!response.empty()) {
            SendHeader(conn_fd, response.size());
            SendData(conn_fd, response.data(), response.size());
        }

        close(conn_fd);
    }
}
