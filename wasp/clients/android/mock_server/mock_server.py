#!/usr/bin/env python3
import socket
import struct
import time
import random
import numpy as np

import morphling_pb2
import global_api_pb2

def current_time_micros():
    return int(time.time() * 1000000)

def run_server(host: str, port: int, m: int, k: int, n: int):
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)

    msg = morphling_pb2.UMessage()

    msg.head.version = 1
    msg.head.magic_flag = 0x12340987
    msg.head.random_num = random.randint(0, 2**32 - 1)
    msg.head.flow_no = random.randint(0, 2**32 - 1)
    msg.head.session_no = str(current_time_micros())
    msg.head.message_type = global_api_pb2.COMPUTE_GEMM_REQUEST

    gemm_req = msg.body.Extensions[global_api_pb2.compute_gemm_request]
    gemm_req.version = 1
    gemm_req.row = m       
    gemm_req.col = n       
    gemm_req.pivot = 0
    gemm_req.h_dim = k     
    gemm_req.dev_id = 0
    gemm_req.oid = 0
    gemm_req.timestamp = current_time_micros()

    gemm_req.matrix_a.offset = 0
    gemm_req.matrix_a.size = A.nbytes
    gemm_req.matrix_b.offset = A.nbytes
    gemm_req.matrix_b.size = B.nbytes

    proto_data = msg.SerializeToString()
    tensor_data = A.tobytes() + B.tobytes()

    header = struct.pack("!II", len(proto_data), len(tensor_data))
    final_message = header + proto_data + tensor_data

    print("Starting server on {}:{}".format(host, port))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        print(f"Python dummy server listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            while True:
                input("Press Enter to send the message...")
                print("Sending message (proto size {} bytes, tensor size {} bytes).".format(len(proto_data), len(tensor_data)))
                conn.sendall(final_message)
                print("Waiting for response...")
                response_header = conn.recv(8)
                response_size = struct.unpack("!I", response_header[:4])[0]
                response_data = conn.recv(response_size)
                print("Received response of size {} bytes.".format(response_size))
                


if __name__ == "__main__":
    config = {"m": 256, "k": 1024, "n": 512}
    run_server("0.0.0.0", 9999, config["m"], config["k"], config["n"])