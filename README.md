# Oasis
[오아시스 소개 영상](https://www.youtube.com/watch?v=p52VW14SraU)

## 라이브러리 설치중 나타날수 있는 에러

ModuleNotFoundError: No module named 'win32file
pywin32가 설치되지 않아 생기는 오류. 

```python
python -m pip install pywin32
```
으로 해결


ERROR: Could not find a version that satisfies the requirement torch==1.9.1+cu102 (from versions: 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2, 1.11.0) ERROR: No matching distribution found for torch==1.9.1+cu102

```python
pip3 install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
** 추가적으로 파이썬 버전이 너무 최신이어도 위의 명령어가 실행되지 않음.
