# python 3.9 설치

     다른 site 정보 참조

# virtualenv 설치

    >python3.9 -m pip install virtualenv

# 가상환경 만들기 

    >mkdir python39_env 
    >python3.9 -m virtualenv python39_env

# 추가 패키지 설치

    >cd python39_env

    (linux)의 경우
    >source bin/activate
    (windows) 경우
    >cd Scripts
    >activate

    (linux, windows)
    >pip install requests
    >pip install numpy
    >pip install pandas
    >deactivate        

# 실행 전 http_clinet.py 수정

        (1) radar_point_cloud_client/dataset/test_460.csv가 존재하는지 확인
        (2) radar_point_cloud_client/main/http_client.py 에서

            line 129) dataset_file_path를 test_460.csv가 존재하는 자신의 절대 경로로 바꾼다.

# Pycharm에서 실행

        radar_point_cloud_client/main/http_client.py 선택해서 Run

# Linux terminal에서 실행

       (1) radar_point_cloud_client/main/http_client.py 파일에서 line 8로 가자

       """ uncommnet this when this lines when this file is executed in [linux] command line"""
       #sys.path.insert(0, os.path.expanduser('~/python39_env/lib/python3.9/site-packages'))
       #sys.path.append(os.path.expanduser('~/radar_point_cloud_client')) <-- 자신의 경로를 확인하자

       위의 두 행의 주석을 제거

       (2) >cd ~/radar_point_cloud_client/main
       (3) >python3.9 http_client.py

# Windows cmd에서 실행

        (1) radar_point_cloud_client/main/http_client.py 파일에서 line 4로 가자

        """ uncommnet this two lines when this file is executed in [windows] command line"""
        #sys.path.insert(0, 'c:/python39_env/lib/site-packages') <- 자신의 경로를 확인하자
        #sys.path.append('d:/projects/radar_point_cloud_client') <- 자신의 경로를 확인하자

        위의 두 행의 주석을 제거

        (2) >cd d:
        (3) >cd projects/radar_point_cloud_client/main
        (4) python3.9 http_client.py           
