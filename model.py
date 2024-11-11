import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 데이터 로드
transform = transforms.ToTensor() #이미지를 pytorch 텐서로 변환해주는 함수(픽셀값을 [0, 0.1]까지 정규화, (높이*너비*채널) 순서를 (채널*높이*너비)로 이미지 차원 재배치, 정수형에서 부동 소수점으로 데어터 타입을 변환)
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)#pytorch에서 제공하는 데이터셋(root=데이터를 저장할 경로 지정, train=훈련 데이터를 다운로드 및 로드할 것인지 나타냄, download=root 디렉토리에 데이터가 없을 경우 자동으로 데이터를 다운로드하는 지 안하는 지를 나타냄, transform=이미지를 변환해줌)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)#train=Flase: 테스트 데이터를 다운로드 및 로드

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)#train_data=학습 데이터, batch_size=데이터를 한번에 몇개 묶어서 가져올 것인지 설정(데이터를 배치단위로 나누는 과정), shuffle=데이터를 섞을지 여부를 결정, **추가 num_workers=데이터를 로드하는 데 사용하는 프로세스 수를 지정
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)#test_data=테스트 데이터

# 단순 모델 정의
class SimpleModel(nn.Module):#pytorch의 기본 신경망인 nn.Module(pytorch가 제공하는 모델 학습, 평가, 매개변수 관리 등 다양한 기능을 사용할 수 있는 기본 신경망 모듈)을 상속받음
    def __init__(self):#모델의 레이어와 파라미터를 정의하는 초기화 함수
        super(SimpleModel, self).__init__()#SimpleModel이 부모 클래스인 nn.Module의 초기화 작업을 실행(super(): 부모 클래스의 기능을 자식 클래스가 사용할 수 있도록 연결해주는 역할)
        self.fc1 = nn.Linear(28 * 28, 10)  # 입력을 28x28에서 10개의 클래스로 매핑(여기서 fc1은 784개의 압력에 대한 가중치 및 편향을 가지고 있음)

    def forward(self, x):#메서드 정의(x: 입력데이터(일반적으로 배치 단위의 이미지 데이터)
        x = x.view(-1, 28 * 28)  # Flatten(이미지를 1차원 배열로 변환)
        x = self.fc1(x)#x에 각 클래스에 대한 예측점수가 들어감(점수는 양일 수도 음일 수도 있다)
        return x

# 모델 초기화
model = SimpleModel()#클래스 정의

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()#손실함수 정의(교차엔트로피 오차)
optimizer = optim.SGD(model.parameters(), lr=0.00001)#옵티마이저 정의(SGD), lr: 학습률

# 훈련
def train(model, train_loader, criterion, optimizer, epochs=1):#함수 정의(model: 학습할 모델, criterion: 손실함수, optimizer: 옵티마이저, epochs: 전체 데이터셋을 학습하는 반복횟수)
    model.train()#모델을 학습모드로 설정(배치 정규화(미니배치의 평균과 분산을 이용)드롭아웃 활성화)
    for epoch in range(epochs):#전체 데이터셋 반복 횟수
        for images, labels in train_loader:#images: 입력 이미지 데이터, labels: 실제 레이블
            outputs = model(images)#모델에 넣어 예측 결과를 생성함
            loss = criterion(outputs, labels)#손실함수 계산
            optimizer.zero_grad()#이전 배치에서 계산된 기울기를 초기화함
            loss.backward()#역전파 알고리즘을 사용하여손실함수의 기울기를 계산
            optimizer.step()#계산된 기울기를 사용하여 모델의 가중치를 업데이트함(SGD)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')#epoch: 전체 데이터셋이 반복된 횟수, loss.item: 손실함수의 결과

# 평가
def evaluate(model, test_loader):#model: 평가할 모델, test_loader: 테스트 데이터셋
    model.eval()#평가 모드로 전환(드롭아웃 비활성화, 배치 정규화(전체 데이터셋의 평균과 분산 이용) 활성화)
    correct, total = 0, 0#correct: 올바르게 예측한 샘플 개수, total: 전체 샘플 수
    with torch.no_grad():#기울기 계산 비활성화
        for images, labels in test_loader:#images:모델의 입력 데이터, labels: 입력데이터에 대한 실제 정답
            outputs = model(images)#모델에 입력 이미지를 넣어서 예측 결과를 얻음
            _, predicted = torch.max(outputs, 1)#outputs: 모델의 예측 결과, 1: 최대값을 찾을 차원, _: 무시한다라는 의미, predicted: 최대값을 가니는 클래스의 인덱스
            total += labels.size(0)#현재 배치의 샘플 수 더하기
            correct += (predicted == labels).sum().item()#올바르게 예측한 샘플의 수를 correct에 더함
    print(f'Accuracy: {100 * correct / total:.2f}%')#전체 샘플 중 올바르게 예측한 샘플의 비율

# 실행
train(model, train_loader, criterion, optimizer, epochs=1)#훈련 함수 실행
evaluate(model, test_loader)#평가 함수 실행