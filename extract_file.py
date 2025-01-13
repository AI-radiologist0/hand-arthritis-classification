import os
import pandas as pd
import shutil
import argparse
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted

def generate_excel_with_checkboxes(base_dir, output_excel):
    """
    폴더와 이미지 파일 구조를 엑셀로 저장하고 체크박스를 추가합니다.

    Args:
        base_dir (str): 기준 디렉토리 경로.
        output_excel (str): 저장할 엑셀 파일 경로.
    """
    data = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            images = [
                img for img in os.listdir(folder_path)
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
            ]
            images = natsorted(images)
            for image in images:
                data.append({"Folder": folder, "Image": image, "Full Path": os.path.join(folder_path, image)})

    # 데이터프레임 생성
    df = pd.DataFrame(data)
    df["Checked"] = False  # 체크박스 열 추가

    # 엑셀 저장
    df.to_excel(output_excel, index=False)
    print(f"Excel file with checkboxes saved to: {output_excel}")


def filter_checked_images(excel_file, output_dir):
    """
    엑셀 파일에서 'Checked' 열이 True인 이미지만 필터링하여 복사.

    Args:
        excel_file (str): 체크박스 열이 포함된 엑셀 파일 경로.
        output_dir (str): 체크된 이미지를 저장할 디렉토리 경로.
    """
    if not os.path.exists(excel_file):
        print("Excel file not found. Skipping filtering.")
        return None

    # 엑셀 파일 읽기
    df = pd.read_excel(excel_file)

    if "Checked" not in df.columns:
        print("The Excel file does not contain a 'Checked' column. Skipping filtering.")
        return None

    # 체크된 이미지 필터링
    checked_images = df[df["Checked"] == True]

    if checked_images.empty:
        print("No checked images found. Skipping filtering.")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 이미지 복사
    for _, row in checked_images.iterrows():
        source = row["Full Path"]
        destination = os.path.join(output_dir, os.path.basename(source))
        shutil.copy(source, destination)
        print(f"Copied: {source} -> {destination}")

    print(f"Checked images copied to: {output_dir}")


class FilteredImageDataset(Dataset):
    """
    필터링된 이미지만 로드하는 데이터셋.
    """
    def __init__(self, base_dir, excel_file=None, transform=None):
        """
        Args:
            base_dir (str): 기준 디렉토리 경로.
            excel_file (str): 체크박스 열이 포함된 엑셀 파일 경로.
            transform (callable, optional): 이미지 변환 함수.
        """
        self.images = []
        self.transform = transform

        if excel_file and os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
            if "Checked" in df.columns:
                self.images = df[df["Checked"] == True]["Full Path"].tolist()
            else:
                print("The Excel file does not contain a 'Checked' column. Loading all images.")
        else:
            print("No valid Excel file provided. Loading all images.")
            for folder in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder)
                if os.path.isdir(folder_path):
                    self.images.extend([
                        os.path.join(folder_path, img) for img in os.listdir(folder_path)
                        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
                    ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        # 필요시 이미지 로드 및 변환
        return image_path


def main():
    parser = argparse.ArgumentParser(description="Image Management Tool with Excel Integration")
    subparsers = parser.add_subparsers(dest="command", help="Choose a command to execute")

    # 엑셀 생성 명령어
    parser_generate = subparsers.add_parser("generate", help="Generate an Excel file with folder and image structure")
    parser_generate.add_argument("--base_dir", required=True, help="Base directory to scan for folders and images")
    parser_generate.add_argument("--output_excel", required=True, help="Path to save the Excel file")

    # 체크된 이미지 필터링 명령어
    parser_filter = subparsers.add_parser("filter", help="Filter and copy checked images")
    parser_filter.add_argument("--excel_file", required=True, help="Excel file with 'Checked' column")
    parser_filter.add_argument("--output_dir", required=True, help="Directory to save the filtered images")

    # 데이터 로더 테스트 명령어
    parser_dataloader = subparsers.add_parser("dataloader", help="Load filtered images using DataLoader")
    parser_dataloader.add_argument("--base_dir", required=True, help="Base directory to scan for folders and images")
    parser_dataloader.add_argument("--excel_file", help="Excel file with 'Checked' column")

    args = parser.parse_args()

    if args.command == "generate":
        generate_excel_with_checkboxes(args.base_dir, args.output_excel)
    elif args.command == "filter":
        filter_checked_images(args.excel_file, args.output_dir)
    elif args.command == "dataloader":
        # dataset = FilteredImageDataset(base_dir=args.base_dir, excel_file=args.excel_file)
        # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        print("Loaded Images:")
        for batch in dataloader:
            print(batch)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
