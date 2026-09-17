import csv
import email
import imaplib
from email.header import decode_header

from .config import DATA_CSV, RAW_CRAWL_CSV


def crawl_subjects():
    """IMAP으로 받은편지함 제목을 긁어서 RAW_CRAWL_CSV에 저장.

    NOTE: 원본 노트북 그대로, 긁어온 제목은 실제 스팸/중요 여부 검증 없이
    전부 label=0으로 저장됨. 학습 데이터에 섞어 쓰기 전에 라벨링 별도 필요.
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
            writer.writerow([0, subject])

    print(f"메일 제목 리스트를 '{RAW_CRAWL_CSV}'에 저장했습니다.")


def fix_encoding():
    """RAW_CRAWL_CSV(euc-kr로 저장된 경우 대비)를 utf-8 DATA_CSV로 변환."""
    try:
        with open(RAW_CRAWL_CSV, 'r', encoding='euc-kr') as infile:
            content = infile.read()
        with open(DATA_CSV, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    crawl_subjects()
    fix_encoding()
