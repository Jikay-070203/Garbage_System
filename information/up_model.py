from huggingface_hub import upload_folder
# Thư mục chứa model
upload_folder(
    folder_path=r"D:\SourceCode\ProGabage\system\up",  
    path_in_repo="",  #upload all
    repo_id="hoanguyenthanh07/gar_system_onnx", 
    repo_type="model",  
    token=""
)