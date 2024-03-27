import os
import re
import subprocess


def is_port_in_use(port):
  """Checks if a specific port is already in use."""
  result = subprocess.run(['netstat', '-atlpn'], capture_output=True, text=True)
  for line in result.stdout.splitlines():
    match = re.search(r'\d+/' + str(port), line)  # Look for digits followed by port number
    if match:
      return True
  return False


def close_port(port):
    """Closes a port by terminating the process using it."""
    if not is_port_in_use(port):
        print(f"Port {port} is not in use.")
        return

    # Find process ID using netstat
    result = subprocess.run(['netstat', '-atlpn'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if str(port) in line:
            pid = int(line.split()[6])  # Assuming PID is in 7th position
            os.kill(pid, 9)  # Terminate process with SIGKILL
            print(f"Closed port {port}.")
            return


def kill_thread():
    # doesn't work
    try:
        print("Closing open3d port connection...")
        from psutil import process_iter
        from signal import SIGKILL

        for proc in process_iter():
            for conns in proc.get_connections(kind='inet'):
                if conns.laddr[1] == 8888:
                    proc.send_signal(SIGKILL)
                    continue
    except:
        pass


if __name__ == '__main__':
    # usage
    port = 8888
    close_port(port)
    # Now you can launch your application on port 8888
