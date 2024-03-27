import psutil


def is_port_in_use(port):
  """Checks if a specific port is already in use."""
  for process in psutil.net_connections():
    if process.laddr[1] == port:  # Access port from the second element of the tuple
      return True
  return False


def free_port(port):
    """Tries to free up the port by terminating the process using it (cautious approach)."""
    if not is_port_in_use(port):
        print(f"Port {port} is already free.")
        return

    for process in psutil.net_connections():
        if process.laddr.port == port:
            print(f"Found process using port {port}: {process.pid} - {process.name}")
            # Be cautious! This terminates the process. Consider checking the process name before termination.
            process.terminate()
            print(f"Process terminated. Port {port} should be free now.")
            return


if __name__ == "__main__":
    if is_port_in_use(8888):
        free_port(8888)
    else:
        print("Port 8888 is available.")
