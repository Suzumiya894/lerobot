import asyncio
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

async def send_msg(writer: asyncio.StreamWriter, msg_dict):
    """
    为消息添加报头，并异步地发送给客户端。
    注意：writer是异步IO对象，不再是socket。
    """
    send_msg_start_time = time.time()

    full_msg = {
        'data': msg_dict,
        'timestamp': time.time()
    }
    
    msg_bytes = packb(full_msg, use_bin_type=True)
    msg_len_header = struct.pack('!I', len(msg_bytes))

    # 使用 writer.write() 发送数据，这是非阻塞的
    writer.write(msg_len_header)
    writer.write(msg_bytes)
    
    # 使用 await writer.drain() 等待缓冲区清空，确保数据已发出
    # 这是异步IO的关键，它会将控制权交还给事件循环
    await writer.drain()
    print(f"发送消息和编码共耗时: {time.time() - send_msg_start_time:.4f}秒")
async def recv_all(reader: asyncio.StreamReader, n: int) -> bytes | None:
    """
    使用 asyncio.StreamReader 异步地、安全地接收 n 个字节。
    """
    try:
        # reader.readexactly(n) 是一个健壮的方法，会等待直到接收到n个字节
        data = await reader.readexactly(n)
        return data
    except asyncio.IncompleteReadError:
        # 如果连接在读取完成前关闭，则会发生此异常
        print("连接在读取数据时被对方关闭。")
        return None

async def recv_msg(reader: asyncio.StreamReader):
    """
    异步地接收完整的消息。
    """
    # 首先异步接收4字节的报头
    raw_msg_len = await recv_all(reader, 4)
    if not raw_msg_len:
        return None
    
    encoding_start_time = time.time()
    msg_len = struct.unpack('!I', raw_msg_len)[0]
    
    # 根据长度异步接收完整的消息体
    data_bytes = await recv_all(reader, msg_len)
    if data_bytes is None:
        return None
        
    msg = unpackb(data_bytes, raw=False)
    encoding_time = time.time() - encoding_start_time
    print(f"解码消息共耗时: {encoding_time:.4f}秒")
    print(f"网络传输消息共耗时: {time.time() - encoding_time - msg['timestamp']:.4f}秒")
    return msg['data']