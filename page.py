import os

def split_exe_to_hex(input_file_path, output_dir="output", skip_all_zero=True):
    """
    将单个exe文件按4K块分割，并将每个块保存为16进制文本文件
    
    参数:
    input_file_path (str): 输入exe文件的路径
    output_dir (str): 输出文件夹路径
    skip_all_zero (bool): 是否跳过全0块
    """
    # 获取文件名（不包含扩展名和路径）
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    
    try:
        # 以二进制模式打开exe文件
        with open(input_file_path, 'rb') as f:
            block_size = 4096  # 4K块大小
            block_number = 1  # 块号从1开始
            written_blocks = 0
            
            # 循环读取文件直到结束
            while True:
                # 读取4K数据
                data = f.read(block_size)
                
                # 如果没有数据，说明已到达文件末尾
                if not data:
                    break
                
                # 检查是否全为0
                if skip_all_zero and all(byte == 0 for byte in data):
                    block_number += 1
                    continue
                
                # 转换为16进制字符串
                hex_data = data.hex()
                
                # 按每16字节(32个字符)分割，并添加换行
                formatted_hex = ''
                for i in range(0, len(hex_data), 32):
                    formatted_hex += hex_data[i:i+32] + '\n'
                
                # 生成输出文件名（格式：原文件名-块号.txt，块号从1开始）
                output_file = os.path.join(output_dir, f"{file_name}-{block_number}.txt")
                
                # 写入16进制数据到文本文件
                with open(output_file, 'w') as out_f:
                    out_f.write(formatted_hex)
                
                block_number += 1
                written_blocks += 1
        
        return written_blocks, block_number - 1  # 返回实际处理的块数
        
    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {e}")
        return 0, 0

def batch_process_exe_files(input_dir, output_dir="output", skip_all_zero=True):
    """
    批量处理目录中的所有exe文件
    
    参数:
    input_dir (str): 包含exe文件的目录路径
    output_dir (str): 输出文件夹路径
    skip_all_zero (bool): 是否跳过全0块
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取目录中所有exe文件
    exe_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.exe')]
    
    if not exe_files:
        print(f"警告: 在目录 {input_dir} 中未找到exe文件")
        return
    
    total_files = len(exe_files)
    total_written_blocks = 0
    total_read_blocks = 0
    
    print(f"开始批量处理 {total_files} 个exe文件...")
    
    # 处理每个exe文件
    for i, exe_file in enumerate(exe_files, 1):
        exe_path = os.path.join(input_dir, exe_file)
        print(f"正在处理文件 {i}/{total_files}: {exe_file}")
        
        written, read = split_exe_to_hex(exe_path, output_dir, skip_all_zero)
        
        total_written_blocks += written
        total_read_blocks += read
        
        print(f"  已处理: 读取 {read} 块，输出 {written} 块")
    
    print(f"\n批量处理完成！")
    print(f"共处理 {total_files} 个文件")
    print(f"共读取 {total_read_blocks} 个块，输出 {total_written_blocks} 个块文件")

if __name__ == "__main__":
    # 用户需要修改此路径为包含exe文件的目录路径
    input_directory = r"C:\malexe"
    # 可选：修改输出目录
    output_directory = r"C:\malpage"
    
    batch_process_exe_files(input_directory, output_directory)    
