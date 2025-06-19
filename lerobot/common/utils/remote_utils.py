import functools
import struct
import time
import msgpack
import torch
import numpy as np


def pack_tensor(obj):
    if isinstance(obj, torch.Tensor):
        # 确保Tensor在CPU上，并转为Numpy
        obj_np = obj.cpu().numpy()
        return {
            b"__tensor__": True,
            b"data": obj_np.tobytes(),
            b"dtype": obj_np.dtype.str,
            b"shape": obj_np.shape,
        }
    elif isinstance(obj, np.ndarray):
        return {
            b"__numpy__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    
    return obj

def unpack_tensor(obj):
    if b"__tensor__" in obj:
        return torch.from_numpy(np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(obj[b"shape"]))
    elif b"__numpy__" in obj:
        return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(obj[b"shape"])
    return obj

packb = functools.partial(msgpack.packb, default=pack_tensor)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_tensor)

def send_msg(sock, msg_dict):
    """
    为消息添加固定长度的报头，然后发送。
    """
    send_msg_start_time = time.time()
    msg_dict = {
        'data': msg_dict,
        'timestamp': time.time()  # 添加时间戳
    }
    # 将消息字典编码为字节
    msg_bytes = packb(msg_dict)
    # 计算消息长度，并打包成一个4字节的整数
    msg_len_header = struct.pack('!I', len(msg_bytes))
    # 发送报头
    sock.sendall(msg_len_header)
    # 发送实际消息
    sock.sendall(msg_bytes)
    print(f"发送消息和编码共耗时: {time.time() - send_msg_start_time:.2f}秒")

def recv_msg(sock):
    """
    接收固定长度的报头以确定消息大小，然后接收完整的消息。
    """
    # 首先接收4字节的报头
    raw_msg_len = recv_all(sock, 4)
    if not raw_msg_len:
        return None
    
    encoding_start_time = time.time()
    # 解包报头以获取消息长度
    msg_len = struct.unpack('!I', raw_msg_len)[0]
    
    # 根据获取的长度接收完整的消息
    data_bytes = recv_all(sock, msg_len)
    if data_bytes is None:
        print(f"Connection closed by opposite side.")
        return None
    msg = unpackb(data_bytes)
    encoding_time = time.time() - encoding_start_time
    print(f"解码消息共耗时: {encoding_time:.2f}秒")
    print(f"网络传输消息共耗时: {time.time() - encoding_time - msg['timestamp']:.2f}秒")
    return msg['data']

def recv_all(sock, n) -> bytes:
    """
    一个辅助函数，确保从套接字接收到n个字节的数据。
    这是必要的，因为单次recv可能不会返回所有请求的数据。
    """
    data = bytearray()
    while len(data) < n:
        # 从缓冲区接收数据
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data