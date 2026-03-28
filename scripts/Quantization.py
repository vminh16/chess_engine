import os
# Thêm hàm quant_pre_process vào import
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

def quantize_model(input_path, output_path):
    print(f"1. Pre-processing model: {input_path}")
    
    # Tạo tên file tạm
    preprocessed_path = input_path.replace(".onnx", "_preprocessed.onnx")
    
    # BƯỚC QUAN TRỌNG: Chạy pre-process để sửa lỗi Shape Inference
    try:
        quant_pre_process(
            input_model_path=input_path,
            output_model_path=preprocessed_path,
            skip_optimization=False 
        )
        print("Pre-processing complete.")
    except Exception as e:
        print(f"Warning: Pre-processing failed ({e}). Trying to quantize directly...")
        preprocessed_path = input_path # Nếu lỗi thì dùng file gốc (thường sẽ fail tiếp nhưng cứ try)

    print(f"2. Quantizing model...")
    
    # Nén file đã được pre-process (không phải file gốc)
    quantize_dynamic(
        model_input=preprocessed_path,  # <--- Dùng file đã sửa lỗi
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"Quantization complete! Saved to: {output_path}")
    
    # Dọn dẹp file tạm nếu muốn
    # if os.path.exists(preprocessed_path):
    #     os.remove(preprocessed_path)

    # So sánh kích thước
    size_fp32 = os.path.getsize(input_path) / (1024 * 1024)
    size_int8 = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Original Size: {size_fp32:.2f} MB")
    print(f"Quantized Size: {size_int8:.2f} MB")
    print(f"Reduction: {size_fp32 / size_int8:.2f}x")

if __name__ == "__main__":
    input_model = "model/param_model/PhantomChessNet.onnx"
    output_model = "model/param_model/PhantomChessNet_int8.onnx"
    
    quantize_model(input_model, output_model)