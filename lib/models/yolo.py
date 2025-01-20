# -----------------------------------------------------------
# YOLOv5 Wrapper with Ultralytics Library
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
# -----------------------------------------------------------
from ultralytics import YOLO


class YOLOv5:
    def __init__(self, model_name='yolov5s', pretrained=True, num_classes=80):
        """
        YOLOv5 모델 클래스 (Ultralytics 활용)

        Args:
            model_name (str): 사용할 YOLOv5 모델의 이름 (e.g., 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            pretrained (bool): 사전 학습된 가중치 사용 여부
            num_classes (int): 출력 클래스 수
        """
        if pretrained:
            # Ultralytics YOLO 모델 로드
            self.model = YOLO(f"{model_name}.pt")
        else:
            # YAML 파일로 새로운 모델 정의
            self.model = YOLO(f"{model_name}.yaml")

        # 클래스 수 업데이트 (필요시 수정)
        if num_classes != self.model.model[-1].nc:
            self.model.model[-1].nc = num_classes  # 클래스 수 업데이트
            self.model.names = [str(i) for i in range(num_classes)]  # 클래스 이름 업데이트

    def train(self, data, epochs=50, imgsz=640, batch=16, device=0):
        """
        모델 학습

        Args:
            data (str): 데이터셋 경로 (data.yaml 파일)
            epochs (int): 학습 에포크 수
            imgsz (int): 입력 이미지 크기
            batch (int): 배치 크기
            device (int or str): 학습에 사용할 디바이스 (GPU ID 또는 'cpu')
        """
        self.model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device)

    def val(self, data, batch=16, device=0):
        """
        모델 검증

        Args:
            data (str): 데이터셋 경로 (data.yaml 파일)
            batch (int): 배치 크기
            device (int or str): 검증에 사용할 디바이스 (GPU ID 또는 'cpu')

        Returns:
            dict: 검증 결과 (mAP, Precision, Recall 등)
        """
        return self.model.val(data=data, batch=batch, device=device)

    def predict(self, source, save=False, device=0):
        """
        이미지 또는 비디오에서 객체를 탐지합니다.

        Args:
            source (str): 입력 데이터 경로 (이미지, 비디오, 폴더 등)
            save (bool): 탐지 결과를 저장할지 여부
            device (int or str): 추론에 사용할 디바이스 (GPU ID 또는 'cpu')

        Returns:
            list: 탐지된 결과
        """
        results = self.model.predict(source=source, save=save, device=device)
        return results

    def save_checkpoint(self, checkpoint_path):
        """
        학습된 가중치를 저장합니다.

        Args:
            checkpoint_path (str): 저장할 경로
        """
        self.model.save(checkpoint_path)

    def load_pretrained(self, checkpoint_path):
        """
        저장된 체크포인트에서 가중치를 로드합니다.

        Args:
            checkpoint_path (str): 체크포인트 경로
        """
        self.model = YOLO(checkpoint_path)
