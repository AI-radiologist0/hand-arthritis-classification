# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import json
import os
import argparse
from ultralytics import YOLO
import cv2

# Argument Parser 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Foot Region Detector using YOLO")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the output JSON file")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to the YOLO model file")
    parser.add_argument("--conf_thresh", type=float, default=0.9, help="Confidence threshold for YOLO detection")
    return parser.parse_args()

# JSON 데이터 생성 함수
def generate_vgg_input_json(data, yolo_model, conf_thresh, output_json):
    output_data = {"data": []}

    for record in data["data"]:
        file_path = record["file_path"]
        patient_id = record["patient_id"]
        diagnosis_class = record["class"]

        # 이미지 로드
        if os.path.exists(file_path):
            img = cv2.imread(file_path)
            if img is None:
                print(f"이미지를 읽을 수 없습니다: {file_path}")
                continue

            # YOLO 모델로 추론
            results = yolo_model(img)
            has_valid_box = False

            for result in results:
                for box in result.boxes:
                    if box.conf > conf_thresh:
                        has_valid_box = True
                        break

                if has_valid_box:
                    break

            # 조건에 맞는 이미지만 JSON에 추가
            if has_valid_box:
                output_data["data"].append({
                    "file_path": file_path,
                    "patient_id": patient_id,
                    "class": diagnosis_class
                })

    # JSON 파일 저장
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)

# 메인 실행부
def main():
    args = parse_args()

    # JSON 파일 읽기
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    # YOLO 모델 로드
    yolo_model = YOLO(args.yolo_model)

    # VGG 입력용 JSON 생성
    generate_vgg_input_json(data, yolo_model, args.conf_thresh, args.output_json)

if __name__ == "__main__":
    main()
