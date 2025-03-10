import os
import platform
from datetime import datetime

def get_file_times_windows(file_path: str):
    create_time = os.path.getctime(file_path)  # 실제 생성일
    modify_time = os.path.getmtime(file_path)  # 수정일
    return (datetime.fromtimestamp(create_time), 
            datetime.fromtimestamp(modify_time))

def get_file_times_macos(file_path: str):
    """
    macOS 전용: st_birthtime(실제 생성일), st_mtime(수정일)
    """
    stat_info = os.stat(file_path)
    birth_time = stat_info.st_birthtime
    modify_time = stat_info.st_mtime
    return (datetime.fromtimestamp(birth_time),
            datetime.fromtimestamp(modify_time))

def get_file_times_linux(file_path: str):
    """
    Linux: 정확한 생성일 개념 없음. st_ctime( inode change time ), st_mtime(수정일)
    """
    stat_info = os.stat(file_path)
    ctime = stat_info.st_ctime
    mtime = stat_info.st_mtime
    return (datetime.fromtimestamp(ctime),
            datetime.fromtimestamp(mtime))

def get_file_times(file_path: str):
    """
    현재 OS에 맞춰 만든 날짜(생성)와 수정 날짜(수정)를 반환한다.
    Windows -> getctime() & getmtime()
    macOS   -> st_birthtime & st_mtime
    Linux   -> st_ctime & st_mtime
    """
    system_name = platform.system()

    # 시스템별 OS 라벨 지정
    if system_name == 'Darwin':
        os_label = 'Mac'
        return get_file_times_macos(file_path), os_label
    elif system_name == 'Windows':
        os_label = 'Windows'
        return get_file_times_windows(file_path), os_label
    elif system_name == 'Linux':
        os_label = 'Linux'
        return get_file_times_linux(file_path), os_label
    else:
        # 그 외 OS는 Linux 로직 사용(ctime), 필요 시 수정 가능
        os_label = system_name
        return get_file_times_linux(file_path), os_label

if __name__ == "__main__":
    file_path = "data/Exception_department.csv"
    (created_dt, modified_dt), os_kind = get_file_times(file_path)

    print(f"[OS 종류]     : {os_kind}")
    print(f"[파일 경로]   : {file_path}")
    print(f"[만든 날짜]   : {created_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[수정 날짜]   : {modified_dt.strftime('%Y-%m-%d %H:%M:%S')}")