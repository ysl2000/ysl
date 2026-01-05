import os
import subprocess
import csv
from typing import Dict, List, Any

# ===================== 配置项 =====================
# 标注文件路径
LABELED_CSV = "labeled_process.csv"
# 标注文件编码（直接指定为Windows默认的GBK）
CSV_ENCODING = "gbk"  # 若仍乱码，可改为"utf-8-sig"
# 内存镜像文件目录
MEMORY_DIR = r"C:\Memory image file"
# Volatility2相关配置
VOL_PATH = r"C:\python2.7\python.exe"  # Python2.7路径
VOL_SCRIPT = r"C:\volatility2\vol.py"  # Volatility2的vol.py路径
# 系统配置文件（需根据镜像实际版本修改）
VOL_PROFILE = "Win7SP1x86_23418"
# 输出数据集文件名
OUTPUT_DATASET = "api_sequences_dataset.csv"
# DLL-API关联映射
DLL_API_MAP = {
    "Crypt32.dll": "CryptEncrypt",
    "Ws2_32.dll": "connect",
    "Kernel32.dll": "CreateRemoteThread",
    "User32.dll": "GetMessageW"
}
# 恶意/良性典型API链
MALICIOUS_CHAIN = [
    "VirtualAlloc", "NtProtectVirtualMemory", "connect", "send", "CreateRemoteThread", "RegSetValueEx"
]
BENIGN_CHAIN = [
    "CreateFileW", "ReadFile", "WriteFile", "GetMessageW", "DispatchMessageW"
]

# ===================== 痕迹提取函数 =====================
def extract_traces(img_filename: str, pid: int) -> Dict[str, List[Any]]:
    img_path = os.path.join(MEMORY_DIR, img_filename)
    traces = {
        "stack_traces": [],
        "mem_traces": [],
        "dll_traces": [],
        "thread_traces": []
    }

    # 1. 提取调用栈痕迹
    try:
        cmd_stack = [
            VOL_PATH, VOL_SCRIPT, "-f", img_path, "--profile", VOL_PROFILE, "stack", "-p", str(pid)
        ]
        result = subprocess.run(
            cmd_stack, capture_output=True, text=True, check=True, encoding="gbk"
        )
        stack_lines = result.stdout.strip().split("\n")
        stack_level = 0
        api_candidates = ["connect", "send", "VirtualAlloc", "CreateRemoteThread", "RegSetValueEx"]
        for line in stack_lines:
            line = line.strip()
            matched_api = [api for api in api_candidates if api in line]
            if matched_api:
                traces["stack_traces"].append({
                    "api": matched_api[0],
                    "stack_level": stack_level,
                    "return_addr": "0x0"
                })
                stack_level += 1
    except Exception as e:
        print(f"警告：{img_filename} PID:{pid} 栈痕迹提取失败：{str(e)}")

    # 2. 提取内存页痕迹
    try:
        cmd_vad = [
            VOL_PATH, VOL_SCRIPT, "-f", img_path, "--profile", VOL_PROFILE, "vadinfo", "-p", str(pid)
        ]
        result = subprocess.run(
            cmd_vad, capture_output=True, text=True, check=True, encoding="gbk"
        )
        vad_lines = result.stdout.strip().split("\n")
        for line in vad_lines:
            if "PAGE_EXECUTE_READWRITE" in line or "RWX" in line:
                traces["mem_traces"].append({
                    "api": "VirtualAlloc",
                    "timestamp": "2024-01-01 00:00:00",
                    "event": "分配可执行内存"
                })
                break
    except Exception as e:
        print(f"警告：{img_filename} PID:{pid} 内存痕迹提取失败：{str(e)}")

    # 3. 提取DLL加载痕迹
    try:
        cmd_dll = [
            VOL_PATH, VOL_SCRIPT, "-f", img_path, "--profile", VOL_PROFILE, "dlllist", "-p", str(pid)
        ]
        result = subprocess.run(
            cmd_dll, capture_output=True, text=True, check=True, encoding="gbk"
        )
        dll_lines = result.stdout.strip().split("\n")
        for line in dll_lines:
            line_lower = line.lower()
            for dll_name, api in DLL_API_MAP.items():
                if dll_name.lower() in line_lower:
                    traces["dll_traces"].append({
                        "dll": dll_name,
                        "load_time": "2024-01-01 00:00:00",
                        "related_api": api
                    })
    except Exception as e:
        print(f"警告：{img_filename} PID:{pid} DLL痕迹提取失败：{str(e)}")

    # 4. 提取线程痕迹
    try:
        cmd_thread = [
            VOL_PATH, VOL_SCRIPT, "-f", img_path, "--profile", VOL_PROFILE, "threads", "-p", str(pid)
        ]
        result = subprocess.run(
            cmd_thread, capture_output=True, text=True, check=True, encoding="gbk"
        )
        thread_lines = result.stdout.strip().split("\n")
        for line in thread_lines:
            if "RemoteThread" in line or "CreateRemoteThread" in line:
                traces["thread_traces"].append({
                    "api": "CreateRemoteThread",
                    "timestamp": "2024-01-01 00:00:00",
                    "event": "创建远程线程"
                })
                break
    except Exception as e:
        print(f"警告：{img_filename} PID:{pid} 线程痕迹提取失败：{str(e)}")

    return traces

# ===================== 序列拼接函数 =====================
def assemble_sequence(traces: Dict[str, List[Any]], label: int) -> str:
    stack_apis = sorted(traces["stack_traces"], key=lambda x: x["stack_level"])
    base_apis = [item["api"] for item in stack_apis if item["api"]]
    mem_apis = [item["api"] for item in traces["mem_traces"] if item["api"]]
    dll_apis = [item["related_api"] for item in traces["dll_traces"] if item["related_api"]]
    thread_apis = [item["api"] for item in traces["thread_traces"] if item["api"]]

    typical_chain = MALICIOUS_CHAIN if label == 1 else BENIGN_CHAIN
    all_apis = list(set(base_apis + mem_apis + dll_apis + thread_apis))
    final_sequence = []
    for api in typical_chain:
        if api in all_apis:
            final_sequence.append(api)
            all_apis.remove(api)
    final_sequence.extend(all_apis)

    return ",".join(final_sequence) if final_sequence else ""

# ===================== 主函数 =====================
def main():
    dataset = []
    # 读取标注文件（直接用GBK编码）
    try:
        with open(LABELED_CSV, "r", encoding=CSV_ENCODING) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    img_filename = row["镜像文件名"]
                    pid = int(row["PID"])
                    label = int(row["标签"])
                except (KeyError, ValueError) as e:
                    print(f"跳过无效行：{str(e)}")
                    continue

                print(f"\n处理：{img_filename} PID:{pid} 标签:{label}")
                traces = extract_traces(img_filename, pid)
                api_sequence = assemble_sequence(traces, label)
                print(f"  生成序列：{api_sequence}")

                dataset.append({
                    "镜像文件名": img_filename,
                    "PID": pid,
                    "api_sequence": api_sequence,
                    "label": label
                })
    except UnicodeDecodeError:
        print(f"错误：{LABELED_CSV}编码不是{CSV_ENCODING}，请将CSV_ENCODING改为'utf-8-sig'重试")
        return

    # 写入数据集
    with open(OUTPUT_DATASET, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["镜像文件名", "PID", "api_sequence", "label"])
        writer.writeheader()
        writer.writerows(dataset)

    print(f"\n处理完成！已生成 {OUTPUT_DATASET}，共 {len(dataset)} 条记录")

if __name__ == "__main__":
    main()
