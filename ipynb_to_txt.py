import json
import sys
import os

def convert_ipynb_to_txt(ipynb_path, txt_path):
    """
    Converts a Jupyter Notebook (.ipynb) file to a text (.txt) file,
    including both code and output from each cell.
    """
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp tại {ipynb_path}")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể giải mã JSON từ {ipynb_path}")
        return

    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        for i, cell in enumerate(notebook.get('cells', [])):
            cell_type = cell.get('cell_type')
            txt_file.write(f"--- Cell {i+1}: {cell_type} ---\n\n")

            source = cell.get('source', [])
            if source:
                source_text = ''.join(source) if isinstance(source, list) else source
                if cell_type == 'code':
                    txt_file.write("### CODE:\n")
                else: # markdown or raw
                    txt_file.write("### MARKDOWN/RAW:\n")
                txt_file.write(source_text)
                txt_file.write("\n\n")

            if cell_type == 'code':
                outputs = cell.get('outputs', [])
                if outputs:
                    txt_file.write("### OUTPUT:\n")
                    for output in outputs:
                        output_type = output.get('output_type')
                        if output_type == 'stream':
                            # stdout, stderr
                            text = output.get('text', [])
                            txt_file.write(''.join(text) if isinstance(text, list) else text)
                        elif output_type in ['display_data', 'execute_result']:
                            # rich outputs, results
                            data = output.get('data', {})
                            if 'text/plain' in data:
                                text = data['text/plain']
                                txt_file.write(''.join(text) if isinstance(text, list) else text)
                        elif output_type == 'error':
                            # errors
                            traceback = output.get('traceback', [])
                            txt_file.write('\n'.join(traceback))
                    txt_file.write("\n\n")

    print(f"Chuyển đổi thành công {ipynb_path} sang {txt_path}")

def main():
    # --- CHỈNH SỬA TẠI ĐÂY ---
    # Cung cấp đường dẫn đến file notebook đầu vào của bạn
    input_path = 'Machine_Learning/dataset2.ipynb' 
    
    # (Tùy chọn) Cung cấp đường dẫn cho file output. 
    # Nếu để trống, file .txt sẽ được tạo cùng thư mục với file input.
    output_path = '' 
    # --- KẾT THÚC CHỈNH SỬA ---

    if not input_path:
        print("Lỗi: Vui lòng cung cấp đường dẫn tệp đầu vào trong biến 'input_path'.")
        return

    if not output_path:
        # Tự động tạo tên file output nếu không được cung cấp
        base_name = os.path.basename(input_path)
        dir_name = os.path.dirname(input_path)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{file_name}.txt")

    convert_ipynb_to_txt(input_path, output_path)

if __name__ == '__main__':
    main()
