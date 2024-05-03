import requests

def send_img(img):
    file_path = img
    target_url = 'http://127.0.0.1:5000/estimate' # 서버 주소

    with open(file_path, 'rb') as f:
        files = {'file' : f}
        res = requests.post(target_url, files=files)

    if res.status_code == 200:
        try:
            result = res.json()
            print("Total Repair Cost:", result)
        except ValueError:
            print("Failed to parse JSON response:", res.text)
    else:
        print('Error:', res.text)

# 단일 이미지 테스트
if __name__ == '__main__':
    file_path = 'samples/damage/0000177_sc-153567.jpg' # 이미지 파일 경로
    send_img(file_path)
