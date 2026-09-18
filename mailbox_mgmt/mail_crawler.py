import csv
import email
import imaplib
from email.header import decode_header

from .config import DATA_CSV, RAW_CRAWL_CSV


def crawl_subjects():
    """IMAP으로 받은편지함 제목만 긁어서 RAW_CRAWL_CSV에 저장 (라벨 없음).

    라벨(스팸/중요)은 여기서 정하지 않는다. 실제로 어떤 메일이 스팸인지는
    사람이 봐야 알 수 있어서, 검증 없이 label=0을 박아두던 예전 방식은
    학습 데이터에 노이즈만 섞는 문제가 있었음. 라벨링은 label_subjects()에서
    사람이 하나씩 보고 진행한다.
    """
    subjects = []

    while input("ㄱㄱ?: ") != '0':
        imap_server_input = input("사용할 메일서비스명: ")
        imap_server = f"imap.{imap_server_input}"
        email_user = input("계정 ID: ")
        email_pass = input("password: ")

        mail = imaplib.IMAP4_SSL(imap_server, 993)
        mail.login(email_user, email_pass)
        print("로그인 성공!")

        mail.select("inbox")
        status, messages = mail.search(None, "ALL")
        message_ids = messages[0].split()

        for index, msg_id in enumerate(message_ids, start=1):
            print(f"{index}/{len(message_ids)} 번째 메일 처리 중...")
            status, msg_data = mail.fetch(msg_id, "(RFC822)")

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        try:
                            if encoding == "cseuckr":
                                subject = subject.decode("euc-kr", errors="ignore")
                            elif encoding == "unknown-8bit":
                                subject = subject.decode("utf-8", errors="ignore")
                            else:
                                subject = subject.decode(encoding if encoding else "utf-8")
                        except (UnicodeDecodeError, TypeError):
                            subject = subject.decode("utf-8", errors="ignore")
                    subjects.append(subject)

        mail.close()
        mail.logout()

    with open(RAW_CRAWL_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for subject in subjects:
            writer.writerow([subject])

    print(f"메일 제목 {len(subjects)}개를 '{RAW_CRAWL_CSV}'에 저장했습니다 (아직 라벨 없음).")
    print("label_subjects()로 라벨링해야 학습 데이터(DATA_CSV)에 들어갑니다.")


def fix_encoding():
    """RAW_CRAWL_CSV가 euc-kr로 저장된 환경에서 깨졌을 경우 제자리에서 utf-8로 복구.

    라벨링 전 원시 크롤링 파일에만 적용한다. DATA_CSV로 직접 쓰지 않는 이유는
    label_subjects()를 거치지 않은 데이터가 검증 없이 학습셋에 섞이는 걸
    막기 위함.
    """
    try:
        with open(RAW_CRAWL_CSV, 'r', encoding='euc-kr') as infile:
            content = infile.read()
        with open(RAW_CRAWL_CSV, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
    except Exception as e:
        print(f"Error occurred: {e}")


def label_subjects():
    """RAW_CRAWL_CSV의 제목을 사람이 하나씩 보고 라벨링해서 DATA_CSV에 추가.

    1=중요문서, 0=스팸, s=건너뛰기(애매하면 학습 데이터에 안 넣는 게 나음),
    q=지금까지 라벨링한 것만 저장하고 중단.
    """
    try:
        with open(RAW_CRAWL_CSV, 'r', encoding='utf-8') as f:
            subjects = [row[0] for row in csv.reader(f) if row]
    except FileNotFoundError:
        print(f"'{RAW_CRAWL_CSV}'가 없습니다. crawl_subjects()를 먼저 실행하세요.")
        return

    if not subjects:
        print(f"'{RAW_CRAWL_CSV}'에 라벨링할 제목이 없습니다.")
        return

    labeled = []
    print(f"총 {len(subjects)}개 제목을 라벨링합니다. (1=중요문서, 0=스팸, s=건너뛰기, q=중단)")

    for index, subject in enumerate(subjects, start=1):
        while True:
            answer = input(f"[{index}/{len(subjects)}] \"{subject}\" -> 1/0/s/q: ").strip().lower()
            if answer in ('1', '0'):
                labeled.append((answer, subject))
                break
            if answer == 's':
                break
            if answer == 'q':
                print("라벨링을 중단합니다.")
                _append_labeled(labeled)
                return
            print("1(중요), 0(스팸), s(건너뛰기), q(중단) 중 하나를 입력하세요.")

    _append_labeled(labeled)


def _append_labeled(labeled):
    if not labeled:
        print("추가된 라벨이 없습니다.")
        return
    with open(DATA_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(labeled)
    print(f"{len(labeled)}개 라벨을 '{DATA_CSV}'에 추가했습니다.")


if __name__ == "__main__":
    crawl_subjects()
    fix_encoding()
    label_subjects()
