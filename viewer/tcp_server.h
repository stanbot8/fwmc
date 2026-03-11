#ifndef FWMC_VIEWER_TCP_SERVER_H_
#define FWMC_VIEWER_TCP_SERVER_H_

// Non-blocking TCP server for viewer-to-nmfly communication.
//
// Sends MotorCommand as MSG_STATUS JSON each frame to a connected
// Python nmfly client. Optionally receives MSG_BIO_READINGS for
// proprioceptive feedback.
//
// Uses WinSock2 on Windows, POSIX sockets elsewhere.
// All I/O is non-blocking (polled per frame, never stalls the viewer).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "bridge/protocol.h"
#include "bridge/tcp_bridge.h"
#include "core/motor_output.h"

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  using socket_t = SOCKET;
  constexpr socket_t kInvalidSock = INVALID_SOCKET;
  inline int sock_close(socket_t s) { return closesocket(s); }
  inline int sock_errno() { return WSAGetLastError(); }
  inline bool sock_would_block() {
    int e = WSAGetLastError();
    return e == WSAEWOULDBLOCK;
  }
#else
  #include <arpa/inet.h>
  #include <errno.h>
  #include <fcntl.h>
  #include <netinet/tcp.h>
  #include <sys/select.h>
  #include <sys/socket.h>
  #include <unistd.h>
  using socket_t = int;
  constexpr socket_t kInvalidSock = -1;
  inline int sock_close(socket_t s) { return close(s); }
  inline int sock_errno() { return errno; }
  inline bool sock_would_block() {
    return errno == EWOULDBLOCK || errno == EAGAIN;
  }
#endif

namespace fwmc {

class ViewerTcpServer {
 public:
  ~ViewerTcpServer() { Stop(); }

  bool Start(uint16_t p) {
    if (listen_sock_ != kInvalidSock) return true;  // already running

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
    wsa_init_ = true;
#endif

    listen_sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock_ == kInvalidSock) return false;

    // Allow reuse
    int opt = 1;
    setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));

    // Bind to loopback
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(p);

    if (bind(listen_sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      sock_close(listen_sock_);
      listen_sock_ = kInvalidSock;
      return false;
    }

    if (listen(listen_sock_, 1) != 0) {
      sock_close(listen_sock_);
      listen_sock_ = kInvalidSock;
      return false;
    }

    SetNonBlocking(listen_sock_);
    port_ = p;
    return true;
  }

  void Stop() {
    if (client_sock_ != kInvalidSock) {
      sock_close(client_sock_);
      client_sock_ = kInvalidSock;
    }
    if (listen_sock_ != kInvalidSock) {
      sock_close(listen_sock_);
      listen_sock_ = kInvalidSock;
    }
#ifdef _WIN32
    if (wsa_init_) {
      WSACleanup();
      wsa_init_ = false;
    }
#endif
  }

  // Call once per frame. Accepts new connections, reads incoming data.
  void Poll() {
    if (listen_sock_ == kInvalidSock) return;

    // Accept new client if none connected
    if (client_sock_ == kInvalidSock) {
      fd_set rfds;
      FD_ZERO(&rfds);
      FD_SET(listen_sock_, &rfds);
      timeval tv{0, 0};  // zero timeout = non-blocking

      if (select(static_cast<int>(listen_sock_) + 1, &rfds, nullptr, nullptr, &tv) > 0) {
        sockaddr_in client_addr{};
        int addr_len = sizeof(client_addr);
        socket_t cs = accept(listen_sock_,
                             reinterpret_cast<sockaddr*>(&client_addr),
#ifdef _WIN32
                             &addr_len);
#else
                             reinterpret_cast<socklen_t*>(&addr_len));
#endif
        if (cs != kInvalidSock) {
          SetNonBlocking(cs);
          // TCP_NODELAY for low latency
          int flag = 1;
          setsockopt(cs, IPPROTO_TCP, TCP_NODELAY,
                     reinterpret_cast<const char*>(&flag), sizeof(flag));
          client_sock_ = cs;
        }
      }
    }

    // Try reading from client (drain any incoming messages)
    if (client_sock_ != kInvalidSock) {
      TryRecv();
    }
  }

  bool HasClient() const { return client_sock_ != kInvalidSock; }
  bool IsListening() const { return listen_sock_ != kInvalidSock; }
  uint16_t Port() const { return port_; }

  // Buffer a motor command for batch sending.
  void BufferMotorCommand(const MotorCommand& cmd) {
    protocol::MotorCommand pcmd;
    pcmd.forward_velocity = cmd.forward_velocity;
    pcmd.angular_velocity = cmd.angular_velocity;
    pcmd.approach_drive   = cmd.approach_drive;
    pcmd.freeze           = cmd.freeze;
    motor_buffer_.push_back(pcmd);
  }

  // Send all buffered motor commands as a batch, then clear buffer.
  bool SendMotorBatch() {
    if (client_sock_ == kInvalidSock || motor_buffer_.empty()) return false;

    if (client_version_ >= 1) {
      uint32_t n = static_cast<uint32_t>(motor_buffer_.size());
      uint32_t payload_size = n * sizeof(protocol::MotorCommand);

      protocol::Header hdr;
      hdr.type = static_cast<uint32_t>(protocol::MsgType::kMotorBatch);
      hdr.payload_size = payload_size;

      if (!SendAll(reinterpret_cast<const char*>(&hdr), sizeof(hdr)) ||
          !SendAll(reinterpret_cast<const char*>(motor_buffer_.data()), payload_size)) {
        DisconnectClient();
        motor_buffer_.clear();
        return false;
      }
      motor_buffer_.clear();
      return true;
    }

    // Fallback: send last command only (legacy clients)
    auto last = motor_buffer_.back();
    motor_buffer_.clear();
    MotorCommand cmd;
    cmd.forward_velocity = last.forward_velocity;
    cmd.angular_velocity = last.angular_velocity;
    cmd.approach_drive = last.approach_drive;
    cmd.freeze = last.freeze;
    return SendMotorCommand(cmd);
  }

  // Send motor command as binary MSG_MOTOR (preferred) with JSON fallback.
  bool SendMotorCommand(const MotorCommand& cmd, float sim_time_ms = 0.0f) {
    if (client_sock_ == kInvalidSock) return false;

    if (client_version_ >= 1) {
      // Binary format (v1+): 16 bytes, no parsing needed.
      protocol::MotorCommand pcmd;
      pcmd.forward_velocity = cmd.forward_velocity;
      pcmd.angular_velocity = cmd.angular_velocity;
      pcmd.approach_drive   = cmd.approach_drive;
      pcmd.freeze           = cmd.freeze;

      protocol::Header hdr;
      hdr.type = static_cast<uint32_t>(protocol::MsgType::kMotor);
      hdr.payload_size = sizeof(pcmd);

      if (!SendAll(reinterpret_cast<const char*>(&hdr), sizeof(hdr)) ||
          !SendAll(reinterpret_cast<const char*>(&pcmd), sizeof(pcmd))) {
        DisconnectClient();
        return false;
      }
      return true;
    }

    // Legacy JSON format (v0 / pre-handshake clients).
    char json[256];
    int len = snprintf(json, sizeof(json),
        "{\"motor\":{\"forward_velocity\":%.4f,"
        "\"angular_velocity\":%.4f,"
        "\"approach_drive\":%.4f,"
        "\"freeze\":%.4f},"
        "\"sim_time_ms\":%.1f}",
        cmd.forward_velocity, cmd.angular_velocity,
        cmd.approach_drive, cmd.freeze, sim_time_ms);

    protocol::Header hdr;
    hdr.type = static_cast<uint32_t>(protocol::MsgType::kStatus);
    hdr.payload_size = static_cast<uint32_t>(len);

    if (!SendAll(reinterpret_cast<const char*>(&hdr), sizeof(hdr)) ||
        !SendAll(json, len)) {
      DisconnectClient();
      return false;
    }
    return true;
  }

  // Access received bio readings (if any arrived this frame).
  const std::vector<BioReading>& LastBioReadings() const {
    return last_bio_readings_;
  }

  bool HasNewBioReadings() const { return has_new_bio_; }
  void ClearBioReadings() { has_new_bio_ = false; }

  const protocol::BodyState& LastBodyState() const { return last_body_state_; }
  bool HasNewBodyState() const { return has_new_body_state_; }
  void ClearBodyState() { has_new_body_state_ = false; }
  uint32_t ClientVersion() const { return client_version_; }

 private:
  socket_t listen_sock_ = kInvalidSock;
  socket_t client_sock_ = kInvalidSock;
  uint16_t port_ = 9100;
  bool wsa_init_ = false;
  uint32_t client_version_ = 0;  // 0 = legacy (no HELLO), 1+ = protocol v1

  // Receive buffer for partial reads
  std::vector<uint8_t> recv_buf_;
  std::vector<BioReading> last_bio_readings_;
  protocol::BodyState last_body_state_ = {};
  bool has_new_bio_ = false;
  bool has_new_body_state_ = false;

  // Motor command buffer for batch sending
  std::vector<protocol::MotorCommand> motor_buffer_;

  static void SetNonBlocking(socket_t s) {
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(s, FIONBIO, &mode);
#else
    int flags = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, flags | O_NONBLOCK);
#endif
  }

  bool SendAll(const char* data, int len) {
    int sent = 0;
    while (sent < len) {
      int n = send(client_sock_, data + sent, len - sent, 0);
      if (n <= 0) {
        if (sock_would_block()) continue;  // retry (non-blocking)
        return false;
      }
      sent += n;
    }
    return true;
  }

  void DisconnectClient() {
    if (client_sock_ != kInvalidSock) {
      sock_close(client_sock_);
      client_sock_ = kInvalidSock;
    }
  }

  void TryRecv() {
    // Read available data into recv_buf_
    uint8_t tmp[4096];
    for (;;) {
      int n = recv(client_sock_, reinterpret_cast<char*>(tmp), sizeof(tmp), 0);
      if (n > 0) {
        recv_buf_.insert(recv_buf_.end(), tmp, tmp + n);
      } else if (n == 0) {
        // Client disconnected
        DisconnectClient();
        recv_buf_.clear();
        return;
      } else {
        if (!sock_would_block()) {
          DisconnectClient();
          recv_buf_.clear();
        }
        break;
      }
    }

    // Parse complete messages from recv_buf_
    while (recv_buf_.size() >= sizeof(protocol::Header)) {
      protocol::Header hdr;
      std::memcpy(&hdr, recv_buf_.data(), sizeof(hdr));

      if (hdr.payload_size > 1 << 20) {
        // Bad data, disconnect
        DisconnectClient();
        recv_buf_.clear();
        return;
      }

      size_t total = sizeof(protocol::Header) + hdr.payload_size;
      if (recv_buf_.size() < total) break;  // wait for more data

      // Dispatch by type
      const uint8_t* payload = recv_buf_.data() + sizeof(protocol::Header);
      if (hdr.type == static_cast<uint32_t>(protocol::MsgType::kHelloClient)) {
        // Protocol handshake: client sends version.
        if (hdr.payload_size >= 4) {
          std::memcpy(&client_version_, payload, 4);
        }
        // Reply with server HELLO.
        uint32_t ver = protocol::kVersion;
        protocol::Header reply;
        reply.type = static_cast<uint32_t>(protocol::MsgType::kHelloServer);
        reply.payload_size = 4;
        SendAll(reinterpret_cast<const char*>(&reply), sizeof(reply));
        SendAll(reinterpret_cast<const char*>(&ver), 4);
      } else if (hdr.type == static_cast<uint32_t>(protocol::MsgType::kBioReadings)) {
        last_bio_readings_ = DeserializeBioReadings(payload, hdr.payload_size);
        has_new_bio_ = true;
      } else if (hdr.type == static_cast<uint32_t>(protocol::MsgType::kBodyState)) {
        if (hdr.payload_size >= sizeof(protocol::BodyState)) {
          std::memcpy(&last_body_state_, payload, sizeof(protocol::BodyState));
          has_new_body_state_ = true;
        }
      } else if (hdr.type == static_cast<uint32_t>(protocol::MsgType::kPing)) {
        protocol::Header pong;
        pong.type = static_cast<uint32_t>(protocol::MsgType::kPong);
        pong.payload_size = 0;
        SendAll(reinterpret_cast<const char*>(&pong), sizeof(pong));
      }

      // Remove processed message
      recv_buf_.erase(recv_buf_.begin(),
                      recv_buf_.begin() + static_cast<ptrdiff_t>(total));
    }
  }
};

}  // namespace fwmc

#endif  // FWMC_VIEWER_TCP_SERVER_H_
