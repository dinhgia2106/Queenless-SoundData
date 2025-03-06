import os
import shutil
import random
import argparse
from pathlib import Path
import logging

path = "Path to the dataset"

def setup_logging():
    """Thiết lập logging để theo dõi quá trình thực thi"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def create_directory(directory_path):
    """Tạo thư mục nếu chưa tồn tại"""
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Đã tạo thư mục: {path}")
    return path

def split_and_organize_dataset(root_dir, categories, split_ratio=(0.7, 0.1, 0.2), seed=42, dry_run=False):
    """
    Chia và tổ chức dữ liệu ảnh thành các tập train, val, test
    
    Args:
        root_dir: Thư mục gốc chứa dữ liệu
        categories: Danh sách các nhãn (thư mục con)
        split_ratio: Tỷ lệ chia tập train, val, test
        seed: Seed cho việc phân chia ngẫu nhiên
        dry_run: Nếu True, chỉ hiển thị thao tác mà không thực hiện
    
    Returns:
        Dict chứa thống kê về số lượng tệp đã di chuyển
    """
    root_dir = Path(root_dir)
    random.seed(seed)
    
    # Kiểm tra tỷ lệ phân chia
    if sum(split_ratio) != 1.0:
        raise ValueError("Tổng tỷ lệ phân chia phải bằng 1.0")
    
    # Tạo cấu trúc thư mục đích
    train_dir = create_directory(root_dir / "train")
    val_dir = create_directory(root_dir / "val")
    test_dir = create_directory(root_dir / "test")
    
    stats = {"train": 0, "val": 0, "test": 0}
    
    # Xử lý từng nhãn
    for category in categories:
        category_dir = root_dir / category
        
        # Kiểm tra thư mục category có tồn tại không
        if not category_dir.exists():
            logging.warning(f"Thư mục {category_dir} không tồn tại. Bỏ qua.")
            continue
        
        # Tạo các thư mục đích cho category này
        train_category_dir = create_directory(train_dir / category)
        val_category_dir = create_directory(val_dir / category)
        test_category_dir = create_directory(test_dir / category)
        
        # Lấy danh sách các tệp trong thư mục category (chỉ lấy file, không lấy thư mục)
        files = [f for f in category_dir.iterdir() 
                if f.is_file() and not f.name.startswith('.')]
        
        # Nếu đã có thư mục train, val, test con trong category, bỏ qua
        if any(d.is_dir() for d in category_dir.iterdir()):
            nested_dirs = [d.name for d in category_dir.iterdir() if d.is_dir()]
            if set(['train', 'val', 'test']).issubset(set(nested_dirs)):
                # Di chuyển từ cấu trúc cũ (nếu có)
                logging.info(f"Phát hiện cấu trúc cũ trong {category}, di chuyển từ cấu trúc cũ...")
                
                old_train_dir = category_dir / "train"
                old_val_dir = category_dir / "val"
                old_test_dir = category_dir / "test"
                
                # Di chuyển từ old_train_dir vào train_category_dir
                if old_train_dir.exists():
                    for file in old_train_dir.iterdir():
                        if file.is_file():
                            dest_file = train_category_dir / file.name
                            if not dry_run:
                                shutil.copy2(file, dest_file)
                            stats["train"] += 1
                
                # Di chuyển từ old_val_dir vào val_category_dir
                if old_val_dir.exists():
                    for file in old_val_dir.iterdir():
                        if file.is_file():
                            dest_file = val_category_dir / file.name
                            if not dry_run:
                                shutil.copy2(file, dest_file)
                            stats["val"] += 1
                
                # Di chuyển từ old_test_dir vào test_category_dir
                if old_test_dir.exists():
                    for file in old_test_dir.iterdir():
                        if file.is_file():
                            dest_file = test_category_dir / file.name
                            if not dry_run:
                                shutil.copy2(file, dest_file)
                            stats["test"] += 1
                
                continue
        
        # Trộn ngẫu nhiên các tệp
        random.shuffle(files)
        
        # Xác định số lượng tệp cho mỗi tập
        total_files = len(files)
        train_split = int(split_ratio[0] * total_files)
        val_split = int((split_ratio[0] + split_ratio[1]) * total_files)
        
        train_files = files[:train_split]
        val_files = files[train_split:val_split]
        test_files = files[val_split:]
        
        logging.info(f"Category {category}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Di chuyển các tệp vào thư mục tương ứng
        for file in train_files:
            dest_file = train_category_dir / file.name
            if not dry_run:
                shutil.copy2(file, dest_file)
            stats["train"] += 1
        
        for file in val_files:
            dest_file = val_category_dir / file.name
            if not dry_run:
                shutil.copy2(file, dest_file)
            stats["val"] += 1
        
        for file in test_files:
            dest_file = test_category_dir / file.name
            if not dry_run:
                shutil.copy2(file, dest_file)
            stats["test"] += 1
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Chia và tổ chức dữ liệu ảnh thành các tập train, val, test")
    parser.add_argument("--data_dir", type=str, default=path,
                       help="Thư mục gốc chứa dữ liệu")
    parser.add_argument("--categories", type=str, nargs="+", default=["bee", "nobee", "noqueen"],
                       help="Danh sách các nhãn (thư mục con)")
    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.1, 0.2],
                       help="Tỷ lệ chia tập train, val, test")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed cho việc phân chia ngẫu nhiên")
    parser.add_argument("--dry_run", action="store_true",
                       help="Chạy thử, không thực hiện di chuyển tệp")
    
    args = parser.parse_args()
    
    setup_logging()
    logging.info(f"Bắt đầu phân chia dữ liệu từ {args.data_dir}")
    logging.info(f"Các nhãn: {args.categories}")
    logging.info(f"Tỷ lệ phân chia: {args.split_ratio}")
    
    try:
        stats = split_and_organize_dataset(
            args.data_dir, 
            args.categories, 
            args.split_ratio, 
            args.seed,
            args.dry_run
        )
        
        if args.dry_run:
            logging.info("Chế độ chạy thử, không có tệp nào được di chuyển")
        
        logging.info(f"Hoàn thành phân chia dữ liệu!")
        logging.info(f"Tổng số lượng: train={stats['train']}, val={stats['val']}, test={stats['test']}")
    except Exception as e:
        logging.error(f"Lỗi: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())