import psutil


def is_port_in_use(port):
    """Checks if a specific port is already in use."""
    for process in psutil.net_connections():
        if process.laddr[1] == port:
            return True
    return False


def free_port(port):
    """Tries to free up the port by terminating the associated process (cautious approach)."""
    if not is_port_in_use(port):
        print(f"Port {port} is already free.")
        return

    for process in psutil.net_connections():
        if process.laddr[1] == port:
            # print(f"Found process using port {port}: {process.pid} - {process.name}")
            print(f"Found process using port {port}.")
            # Get the process object associated with the connection
            try:
                parent = process.pid  # Assuming the parent process is using the port
                parent_process = psutil.Process(parent)
                parent_process.terminate()
                print(f"Process terminated (PID: {parent}). Port {port} should be free now.")
            except (psutil.NoSuchProcess, PermissionError) as e:
                print(f"Error terminating process: {e}")
            return


if __name__ == "__main__":
    try:
        if is_port_in_use(8888):
            free_port(8888)
        else:
            print("Port 8888 is available.")
    except Exception as err:
        print(err)
