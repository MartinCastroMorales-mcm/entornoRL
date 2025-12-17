import socket
import threading
from queue import Queue
from threading import Event
from enum import StrEnum

class GameControllerTCPServer:

  #_instance = None

  class ServerState(StrEnum):
    INITIAL = "INITIAL"
    LISTENING = "LISTENING"
    CONNECTED = "CONNECTED"

  
  #def __new__(cls):
    ##this singleton stuff does not seem to work
    #print("__new__")
    #if not cls._instance:
      #cls._instance = super().__new__(cls)
    #return cls._instance

  def __init__(self, inbound_queue: Queue, step_allowed: Event):
    self.HEADER = 64
    self.PORT = 0
    self.SERVER_IP = "127.0.0.1"
    self.ADDRESS = (self.SERVER_IP, self.PORT)
    self.FORMAT = "utf-8"
    self.DISCONNECT_MESSAGE = "!DISCONNECT"

    self.currentState = self.ServerState.INITIAL
    #if getattr(self, "_initialized", False):
      #return

    self.ready = Event()
    self.inbound_queue = inbound_queue
    self.step_allowed = step_allowed
    #self._initialization = True



    print("create server init")

    #AF_INET = ipv4
    #SOCK_STREAM = TCP
    self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.server.bind(self.ADDRESS)
      print(f"server binding succesfull to {self.ADDRESS}")
    except:
      print("Error binding to port")
      raise RuntimeError(f"Failed to bind to {self.ADDRESS}")

      
    # dictionario IP -> [] : cliente
    self.client_messages = {}
    self.first_addr = None


  def start(self):
    self.server.listen()
    self.ready.set()
    print(f"[Listening] Server is listening on {self.SERVER_IP}")
    self.currentState = self.ServerState.LISTENING
    print("SERVER STATE -> LISTENING")
    print("BOUND TO:", self.server.getsockname())
    while True:
      print("CALLING accept()")
      conn, addr = self.server.accept()
      print("ACCEPTED:", addr)
      if not self.first_addr:
        self.first_addr = addr
      self.client_messages[addr] = []
      print("addr" + str(addr))
      #print(self.ADDRESS)
      print("create handle_client thread")
      thread = threading.Thread(target=self.handle_client, args=(conn, addr))
      thread.start()
      print(f"[ACTIV CONNECTIONS] {threading.active_count() - 1}")

  def handle_client(self, conn, addr):
    print(f"[NEW CONNECTION, in handle_client] {addr} connected")
    print("this is handle_client")

    connected = True
    self.currentState = self.ServerState.CONNECTED

    while connected:
      try:
        print("waiting for message length", flush=True)
        msg_length = conn.recv(self.HEADER).decode(self.FORMAT)
        if(msg_length):
          msg_length = int(msg_length)
          #print("waiting for message")
          msg = conn.recv(msg_length).decode(self.FORMAT)
          #print("msg received")
          if(msg == self.DISCONNECT_MESSAGE):
            print("disconected")
            connected = False
          self.client_messages[addr].append(msg)

          self.inbound_queue.put(msg)
          print("handle client, wait")
          self.step_allowed.wait()
          self.step_allowed.clear()
          print("handle client, clear")

          conn.sendall("STEP".encode(self.FORMAT))
          #print("step")
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
  

#if __name__ == "__main__":
  #start_server()