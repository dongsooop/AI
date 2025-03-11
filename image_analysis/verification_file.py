# verification_file.py
import os
import platform
from datetime import datetime

def get_file_times_windows(file_path: str):
    """Windows: getctime() → 실제 생성일, getmtime() → 수정일"""
    create_time = os.path.getctime(file_path)
    modify_time = os.path.getmtime(file_path)
    return (datetime.fromtimestamp(create_time), 
            datetime.fromtimestamp(modify_time))

def get_file_times_macos(file_path: str):
    """macOS: st_birthtime(파일 생성), st_mtime(수정)"""
    stat_info = os.stat(file_path)
    birth_time = stat_info.st_birthtime
    modify_time = stat_info.st_mtime
    return (datetime.fromtimestamp(birth_time),
            datetime.fromtimestamp(modify_time))

def get_file_times_linux(file_path: str):
    """
    Linux: '생성 시간' 표준 개념 없음.
    st_ctime 은 inode change time, st_mtime 은 수정 시간.
    """
    stat_info = os.stat(file_path)
    ctime = stat_info.st_ctime
    mtime = stat_info.st_mtime
    return (datetime.fromtimestamp(ctime),
            datetime.fromtimestamp(mtime))

def get_file_times(file_path: str):
    """
    OS 종류에 맞춰 (만든 날짜, 수정 날짜)와 OS 라벨을 반환
    """
    system_name = platform.system()
    if system_name == 'Darwin':  # macOS
        created_dt, modified_dt = get_file_times_macos(file_path)
        os_label = "Mac"
    elif system_name == 'Windows':
        created_dt, modified_dt = get_file_times_windows(file_path)
        os_label = "Windows"
    elif system_name == 'Linux':
        created_dt, modified_dt = get_file_times_linux(file_path)
        os_label = "Linux"
    else:
        # 그 외 OS는 Linux 방식 재사용
        created_dt, modified_dt = get_file_times_linux(file_path)
        os_label = system_name

    return (created_dt, modified_dt), os_label

def is_file_modified(file_path: str):
    """
    파일이 '수정'되었는지 여부를 초 단위 기준으로 판단.
    (macOS/Windows → 실제 생성일, Linux → ctime을 생성일 대용)
    """
    (created_dt, modified_dt), os_label = get_file_times(file_path)
    
    # 초 단위로 환산 (마이크로초 이하 무시)
    created_ts  = int(created_dt.timestamp())
    modified_ts = int(modified_dt.timestamp())

    # 수정 시간이 생성 시간보다 큰(초 단위) 경우 -> 수정됨
    if modified_ts > created_ts:
        return True, created_dt, modified_dt, os_label
    else:
        return False, created_dt, modified_dt, os_label