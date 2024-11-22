import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from omegaconf import DictConfig
import hydra
@hydra.main(config_path='conf', config_name='config', version_base=None)# 설정파일을 로드하고, 해당 설정에 따라 프로그램을 실행하도록 함(config_path: 설정파일이 위치한 디렉토리 경로, config_name: 설정파일 이름, version_base: Hydra의 버전 관리를 위한 인자, None은 가장 최근 버전을 사용한다는 의미
def main(cfg: DictConfig): # cfg: Hydra가 로드한 구성정보를 담고 있는 객체, DictConfig: yaml형식의 구성파일을 파싱하여 Python의 딕셔너리 형태로 변환한 것
    writer = SummaryWriter()# 로그 데이터를 기록하기 위한 모듈에서 제공하는 클래스
    # 데이터 로드
    transform= transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.1, 0)), transforms.ToTensor()])
    """
    transforms.Compose: 여러개의 변환을 사용하기 위한 함수
    transforms.RandomAffine: 데이터를 이미지를 회전, 이동 등을 조작하는 함수이다.
    translate: X축과 Y축의 이동 범위를 지정하는 인자(nn.linear는 작은 이동에도 결과가 민감하게 반응함과 여러번 실험을 거쳐서 그래도 그나마 가장 정확도가 잘 나온 0.1 0.1을 주었다.)
    degrees: 회전 각도(nn.linear는 작은 이동에도 결과가 민감하게 반응하기 때문에 그냥 0을 주었다.)
    transforms.ToTensor(): 이미지를 Pytorch 텐서로 변환함
    """
    #transform=transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)# transform에 transform을 적용해서 변환된 데이터를 가져옴
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())# test데이터는 원본데이터를 가져와야하기때문에 Pytorch 텐서로만 변환함
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 단순 모델 정의
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 10)  # 입력을 28x28에서 10개의 클래스로 매핑

        def forward(self, x):
            x = x.view(-1, 28 * 28)  # Flatten
            x = self.fc1(x)
            return x

    # 모델 초기화
    model = SimpleModel()

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.model.lr) # cfg.model.lr: config.yaml에서 가져온 값

    # 훈련
    def train(model, train_loader, criterion, optimizer, epochs=1):
        model.train()
        for epoch in range(epochs):
            for images, labels in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                writer.add_scalar("Loss/train", loss, epoch)# Tensorboard에 스칼라값을 기록함, 인자는 순서대로 이름, 스칼라값(Y축), 현재 훈련 상태(x축)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')# logging.info 정보수준의 로그를 기록하는 함수

    # 평가
    def evaluate(model, test_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        logging.info(f'Accuracy: {100 * correct / total:.2f}%')

    # 실행
    train(model, train_loader, criterion, optimizer, epochs=cfg.train.epochs)
    evaluate(model, test_loader)
if __name__ == '__main__': # 스크립트가 직접 실행 될때만 main()함수 호출
    main()