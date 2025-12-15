import socket
import threading
import time
from queue import Queue
from threading import Event



class GameControllerTCPServer:

  #_instance = None

  HEADER = 64
  PORT = 5050
  SERVER_IP = "127.0.0.1"
  ADDRESS = (SERVER_IP, PORT)
  FORMAT = "utf-8"
  DISCONNECT_MESSAGE = "!DISCONNECT"

  #def __new__(cls):
    ##this singleton stuff does not seem to work
    #print("__new__")
    #if not cls._instance:
      #cls._instance = super().__new__(cls)
    #return cls._instance

  def __init__(self, inbound_queue: Queue, step_allowed: Event):
    #if getattr(self, "_initialized", False):
      #return

    self.inbound_queue = inbound_queue
    self.step_allowed = step_allowed
    #self._initialization = True

    print("create server tcp init")
    self.host = self.PORT
    self.port = self.SERVER_IP
    #AF_INET = ipv4
    #SOCK_STREAM = TCP
    self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.server.bind(self.ADDRESS)
    except:
      print("Error binding to port")
    # dictionario IP -> [] : cliente
    self.client_messages = {}
    self.first_addr = None


  def start(self):
    self.server.listen()
    print(f"[Listening] Server is listening on {self.SERVER_IP}")
    while True:
      conn, addr = self.server.accept()
      if not self.first_addr:
        self.first_addr = addr
      self.client_messages[addr] = []
      thread = threading.Thread(target=self.handle_client, args=(conn, addr))
      thread.start()
      print(f"[ACTIV CONNECTIONS] {threading.active_count() - 1}")
  
  def handle_client(self, conn, addr):
    print(f"[NEW CONNECTION] {addr} connected")

    connected = True

    while connected:
      try:
        msg_length = conn.recv(self.HEADER).decode(self.FORMAT)
        if(msg_length):
          msg_length = int(msg_length)
          msg = conn.recv(msg_length).decode(self.FORMAT)
          if(msg == self.DISCONNECT_MESSAGE):
            connected = False
          self.client_messages[addr].append(msg)
          #print(f"[{addr}] {msg}")

          self.inbound_queue.put(msg)

          self.step_allowed.wait()
          self.step_allowed.clear()
          ##send the message to the environment
          #with shared_state.lock: #mutex
            #shared_state.action_ready.set()
          #shared_state.action_ready.wait()

          ##wait for the environment to process the action
          #shared_state.step_ready.wait()
          #shared_state.step_ready.clear()

          conn.sendall("STEP".encode(self.FORMAT))
      except Exception as e:
        print("Game close closing thread")
        break
    conn.close()

  def get_message(self, addr):
    if addr:
      return self.client_messages.get(addr, [])
    return self.client_messages.get(self.first_addr, [])

  def send_action(self, action):
      self.client.sendall(str(action).encode())

  def close(self):
      self.client.close()


def start_server():
  TCPServer = GameControllerTCPServer()
  print("Server is starting")
  TCPServer.start()
  

if __name__ == "__main__":
  start_server()