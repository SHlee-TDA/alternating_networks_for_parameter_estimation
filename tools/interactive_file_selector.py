import os
import sys
import termios
import tty
import atexit

def _get_key():
    """Linux 환경에서 키보드 입력을 즉시(Enter 없이) 받아오는 내부 함수"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # 화살표 키는 '\x1b[A', '\x1b[B' 형태로 들어오므로 추가로 읽어줌
        if ch == '\x1b':
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def interactive_file_selector(prompt_msg="파일을 선택하세요:", start_dir=".", use_emoji=True):
    """
    CLI 환경에서 화살표 키로 디렉토리를 탐색하고 파일을 선택하는 모듈
    
    Args:
        prompt_msg (str): 사용자에게 보여줄 안내 문구
        start_dir (str): 탐색을 시작할 초기 경로
        use_emoji (bool): 이모지 사용 여부 (False일 경우 텍스트 아이콘 사용)
        
    Returns:
        str: 선택된 파일의 (초기 경로 기준) 상대 경로
    """
    current_dir = os.path.abspath(start_dir)
    original_dir = current_dir
    selected_index = 0

    # 터미널 상태 복구 보장
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings))

    while True:
        # 1. 현재 경로의 파일/디렉토리 목록 수집
        try:
            items = os.listdir(current_dir)
        except PermissionError:
            items = []

        # 폴더와 파일을 분리하고 알파벳 순 정렬
        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_dir, d))])
        files = sorted([f for f in items if os.path.isfile(os.path.join(current_dir, f))])

        display_list = []
        
        # 상위 디렉토리로 이동 가능 여부 확인 (original_dir보다 상위로 못 가게 제한)
        try:
            rel_path = os.path.relpath(current_dir, original_dir)
            # '..'이 없으면 original_dir과 같거나 하위 경로임
            if not rel_path.startswith('..') and current_dir != original_dir:
                display_list.append(("..", "back"))
        except ValueError:
            # Windows에서 다른 드라이브일 경우 발생 가능
            pass
            
        for d in dirs: display_list.append((d, "dir"))
        for f in files: display_list.append((f, "file"))

        # 인덱스 범위 보정
        if len(display_list) == 0:
            selected_index = 0
        else:
            selected_index = max(0, min(selected_index, len(display_list) - 1))

        # 2. 화면 지우기 및 출력 준비 (ANSI Escape Code 활용)
        sys.stdout.write('\033[2J\033[H') # 화면 클리어 후 커서를 좌측 상단으로
        
        print(f"\033[1;36m{prompt_msg}\033[0m\r") # 안내 문구 (청록색 굵게)
        print(f"현재 경로: \033[4m{os.path.relpath(current_dir, original_dir) if current_dir != original_dir else '.'}\033[0m\r\n")

        # 빈 디렉토리 처리
        if not display_list:
            print("\033[33m(빈 디렉토리입니다. Ctrl+C로 취소)\033[0m\r\n")
            sys.stdout.flush()
            key = _get_key()
            if key == '\x03':
                sys.stdout.write('\033[2J\033[H')
                print("선택이 취소되었습니다.\r")
                sys.exit(0)
            continue

        # 3. 목록 출력
        for i, (name, item_type) in enumerate(display_list):
            # 파일명 sanitize (특수문자 제거)
            display_name = name.replace('\r', '').replace('\n', ' ')
            
            # 아이콘 설정 (이모지 또는 텍스트)
            if use_emoji:
                if item_type == "back": icon = "🔙 "
                elif item_type == "dir": icon = "📁 "
                else: icon = "📄 "
            else:
                if item_type == "back": icon = "[↑] "
                elif item_type == "dir": icon = "[D] "
                else: icon = "[F] "
            
            if i == selected_index:
                # \033[7m: 색상 반전(Highlight) - 깜빡임은 제거 (일부 터미널에서 미지원)
                sys.stdout.write(f"\033[7m> {icon}{display_name}\033[0m\r\n")
            else:
                sys.stdout.write(f"  {icon}{display_name}\r\n")
        
        print(f"\n\033[90m[↑↓: 이동 | Enter: 선택 | Ctrl+C: 취소]\033[0m\r")
        sys.stdout.flush()

        # 4. 사용자 입력 처리
        key = _get_key()
        
        if key == '\x1b[A': # 위 화살표
            selected_index = max(0, selected_index - 1)
        elif key == '\x1b[B': # 아래 화살표
            if len(display_list) > 0:
                selected_index = min(len(display_list) - 1, selected_index + 1)
        elif key in ('\r', '\n'): # Enter 키
            if not display_list: 
                continue
            
            selected_name, selected_type = display_list[selected_index]
            
            if selected_type == "back":
                current_dir = os.path.dirname(current_dir) # 상위 폴더로
                selected_index = 0
            elif selected_type == "dir":
                new_dir = os.path.join(current_dir, selected_name)
                # 심볼릭 링크 순환 참조 방지
                if os.path.islink(new_dir):
                    try:
                        real_path = os.path.realpath(new_dir)
                        if os.path.isdir(real_path):
                            current_dir = real_path
                            selected_index = 0
                    except (OSError, RuntimeError):
                        # 순환 참조 등의 문제 발생 시 무시
                        pass
                else:
                    current_dir = new_dir
                    selected_index = 0
            elif selected_type == "file":
                # 파일 선택 시 화면을 깔끔하게 지우고 상대 경로 반환
                sys.stdout.write('\033[2J\033[H')
                final_path = os.path.relpath(os.path.join(current_dir, selected_name), original_dir)
                return final_path
                
        elif key == '\x03': # Ctrl+C (강제 종료)
            sys.stdout.write('\033[2J\033[H')
            print("선택이 취소되었습니다.\r")
            sys.exit(0)

# (테스트용 단독 실행 코드)
if __name__ == "__main__":
    selected_file = interactive_file_selector(
        "분석할 실험의 Config 또는 가중치 파일을 선택하세요:", 
        start_dir="./results",
        use_emoji=True  # False로 설정하면 텍스트 아이콘 사용
    )
    print(f"\n선택된 파일 경로: {selected_file}")
