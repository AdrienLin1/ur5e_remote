import requests
import numpy as np
import io
import cv2
import zmq
import json
import base64
import time # 补充导入 time 模块

# --- ZMQ 配置 ---
# ZMQ 服务器地址 (对应 Isaac Sim 服务器中绑定的地址)
ZMQ_SERVER_URL = "tcp://192.168.1.100:5555"

# --- 示例数据 ---
# 注意: ZMQ 服务器现在只要求 initial_file 和 action_file
# image_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) # 示例图像数组 (未发送，仅保留定义)
# depth_array = np.full((480, 640), 0.2).astype(np.float32) # 示例深度数组 (未发送，仅保留定义)
# proprio_array = np.random.rand(1,).astype(np.float32) # 示例本体感受数据 (未发送，仅保留定义)

# 动作和初始状态数据（对应原代码中的 .npy 文件内容）
ACTION_DATA = np.array([1, 1, 1, 1, 0, 0, 0], dtype=np.float32)
INITIAL_DATA = np.array([1, 1, 1, 1, 0, 0, 0], dtype=np.float32)


def serialize_numpy(data: np.ndarray) -> str:
    """将 NumPy 数组序列化为 .npy 字节流，并进行 Base64 编码。"""
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def send_data_zmq():
    # 1. 初始化 ZMQ 上下文和请求 (REQ) Socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(ZMQ_SERVER_URL)
    print(f"ZMQ 客户端已连接到 {ZMQ_SERVER_URL}")

    # 2. 序列化数据
    initial_b64 = serialize_numpy(INITIAL_DATA)
    action_b64 = serialize_numpy(ACTION_DATA)

    # 3. 构造 JSON 请求体 (只包含动作和初始状态数据)
    request_data = {
        "initial_file_b64": initial_b64,
        "action_file_b64": action_b64,
    }

    # 4. 发送 JSON 请求
    print("发送 ZMQ 请求...")
    socket.send_json(request_data)

    # 5. 等待并接收 JSON 响应
    try:
        # 设置超时，防止服务端长时间无响应导致客户端卡死
        socket.RCVTIMEO = 1000000  # 10秒超时
        response_data = socket.recv_json()
        
        # 6. 处理服务端返回的结果
        if "error" in response_data:
            print(f"Server error: {response_data['error']}")
        else:
            print("Server response:", response_data)
            # 如果需要，可以解码图像：
            # if 'camera_image_b64' in response_data:
            #     img_data = base64.b64decode(response_data['camera_image_b64'])
            #     # 这里可以进一步处理图像字节流
            
    except zmq.error.Again:
        print(f"Error: ZMQ Request timed out after 10 seconds. Check if server ({ZMQ_SERVER_URL}) is running.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    finally:
        # 清理 ZMQ 资源
        socket.close()
        context.term()


if __name__ == "__main__":
    send_data_zmq()
