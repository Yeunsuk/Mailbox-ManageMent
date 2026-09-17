import email
import imaplib
import json
import socket
import time
from email.header import decode_header

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from .config import PREDICT_MAX_LEN, TKRNN_MODEL, TOKENIZER_JSON
from .metrics import f1_score


def load_artifacts():
    with open(TOKENIZER_JSON, 'r') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)

    model = load_model(TKRNN_MODEL, custom_objects={'f1_score': f1_score})
    return tokenizer, model


def search_with_retry(mail, search_condition, retries=3, delay=5):
    """타임아웃/네트워크 오류 발생 시 재시도하는 IMAP search."""
    attempt = 0
    while attempt < retries:
        try:
            status, messages = mail.search(None, search_condition.encode('utf-8'))
            if status == "OK":
                return messages
            print("검색 실패. 다시 시도 중...")
            time.sleep(delay)
        except (imaplib.IMAP4.abort, socket.error) as e:
            print(f"IMAP 서버 오류 또는 네트워크 오류 발생: {e}. 재시도 중...")
            time.sleep(delay)
        attempt += 1
    raise Exception("검색에 실패했습니다. 재시도 횟수를 초과했습니다.")


def predict_spam(subject, tokenizer, model, max_len=PREDICT_MAX_LEN):
    """메일 제목 기준으로 스팸 여부 예측. 1=중요문서, 0=스팸."""
    processed_subject = subject.lower()
    processed_subject = tokenizer.texts_to_sequences([processed_subject])

    if len(processed_subject[0]) == 0:
        return 0

    processed_subject = pad_sequences(processed_subject, maxlen=max_len)
    prediction = model.predict(processed_subject)

    if prediction[0][0] > 0.5:
        return 1
    return 0


def convert_date(date_str):
    """'2022-01-31' 형식을 IMAP SEARCH가 요구하는 '31-Jan-2022' 형식으로 변환."""
    month_map = {
        '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
        '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
        '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec',
    }

    try:
        year, month, day = date_str.split('-')

        if len(year) != 4 or not year.isdigit():
            raise ValueError("잘못된 연도입니다.")
        if not month.isdigit() or int(month) not in range(1, 13):
            raise ValueError("잘못된 월입니다.")
        if len(day) != 2 or not day.isdigit() or not (1 <= int(day) <= 31):
            raise ValueError("잘못된 날짜입니다.")

        month_name = month_map.get(month)
        if not month_name:
            raise ValueError("잘못된 월입니다.")

        return f"{day}-{month_name}-{year}"

    except ValueError as e:
        return f"오류: {e}"
    except Exception as e:
        return f"날짜 형식이 잘못되었습니다. {str(e)}"


def select_search_condition():
    print("메일 검색 조건을 선택하세요:")
    print("1. 모든 메일 (ALL)")
    print("2. 읽지 않은 메일 (UNSEEN)")
    print("3. 최근 메일 (RECENT)")
    print("4. 읽은 메일 (SEEN)")
    print("5. 답장한 메일 (ANSWERED)")
    print("6. 삭제된 메일 (DELETED)")
    print("7. 초안 메일 (DRAFT)")
    print("8. 중요 표시된 메일 (FLAGGED)")
    print("9. 특정 날짜 이후 받은 메일 (SINCE)")
    print("10. 특정 날짜 이전 받은 메일 (BEFORE)")
    print("11. 특정 날짜에 받은 메일 (ON)")
    print("12. 본문에 특정단어가가 포함된 메일 (BODY)")

    while True:
        search_condition = input("검색 조건을 선택 (1~12): ")

        if search_condition == '1':
            print("모든 메일을 선택")
            return "ALL"
        elif search_condition == '2':
            print("읽지 않은 메일을 선택")
            return "UNSEEN"
        elif search_condition == '3':
            print("최근 메일을 선택")
            return "RECENT"
        elif search_condition == '4':
            print("읽은 메일을 선택")
            return "SEEN"
        elif search_condition == '5':
            print("답장한 메일을 선택")
            return "ANSWERED"
        elif search_condition == '6':
            print("삭제된 메일을 선택")
            return "DELETED"
        elif search_condition == '7':
            print("초안 메일을 선택")
            return "DRAFT"
        elif search_condition == '8':
            print("중요 표시된 메일을 선택")
            return "FLAGGED"
        elif search_condition == '9':
            data = input("날짜를 입력하세요 (예: 2022-01-31): ")
            formatted_date = convert_date(data)
            if "오류" in formatted_date:
                print(formatted_date)
                continue
            print(f'"{formatted_date}" 이후 받은 메일을 선택')
            return f'SINCE "{formatted_date}"'
        elif search_condition == '10':
            data = input("날짜를 입력하세요 (예: 2022-01-31): ")
            formatted_date = convert_date(data)
            if "오류" in formatted_date:
                print(formatted_date)
                continue
            print(f'"{formatted_date}" 이전 받은 메일을 선택')
            return f'BEFORE "{formatted_date}"'
        elif search_condition == '11':
            data = input("날짜를 입력하세요 (예: 2022-01-31): ")
            formatted_date = convert_date(data)
            if "오류" in formatted_date:
                print(formatted_date)
                continue
            print(f'"{formatted_date}"에 받은 메일을 선택')
            return f'ON "{formatted_date}"'
        elif search_condition == '12':
            data = input("특정단어를 입력하세요 : ")
            print(f'"{data}"가 포함된 메일을 선택')
            return f'BODY "{data}"'
        else:
            print("잘못된 입력입니다. 다시 입력하세요.")


def safe_decode_header(header_value):
    decoded_parts = decode_header(header_value)
    decoded_string = ""

    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            try:
                if encoding == "cseuckr":
                    decoded_string += part.decode("euc-kr", errors="ignore")
                elif encoding == "unknown-8bit":
                    decoded_string += part.decode("utf-8", errors="ignore")
                else:
                    decoded_string += part.decode(encoding if encoding else "utf-8")
            except (UnicodeDecodeError, TypeError):
                # NOTE: 원본 노트북 그대로 - 여기서 참조하는 `e`는 정의된 적 없음.
                # 실제로 디코딩 오류가 나면 이 print문 자체가 NameError를 던져서
                # 원래 처리하려던 예외 대신 새 예외로 죽는다. 별도 이슈로 수정 예정.
                print(f"디코딩 오류: {e}")
                decoded_string += part.decode("utf-8", errors="ignore")
        else:
            decoded_string += part

    return decoded_string


def manual_test(tokenizer, model):
    """콘솔에서 제목 하나씩 입력해서 예측 결과를 바로 확인하는 수동 테스트 루프."""
    while True:
        subject = input("메일 제목을 입력하세요 (종료하려면 'exit' 입력): ")
        if subject.lower() == 'exit':
            print("예측을 종료합니다.")
            break

        result = predict_spam(subject, tokenizer, model)
        if result == 1:
            print("이 메일은 중요문서입니다.")
        else:
            print("이 메일은 쓰레기기입니다.")


def run_cleanup():
    """실제 메일함에 로그인해서 예측 결과에 따라 스팸 메일을 휴지통으로 이동/삭제.

    주의:
    - 네이버 계정은 \\Deleted 플래그 + mail.expunge()로 처리되어 즉시 영구 삭제됨
      (휴지통 경유 없음). Gmail은 Trash 라벨 이동이라 상대적으로 안전함.
    - 복구("원상복구 ㄱㄱ?") 여부와 무관하게 expunge()는 항상 호출됨: 복구를
      선택하지 않으면 그대로 영구 삭제되니 사용 전 반드시 인지할 것.
    - 판단 근거는 메일 제목 한 줄뿐이며, 삭제 전 사람이 확인하는 단계가 없음.
    이 동작은 원본 노트북 그대로 옮긴 것이며 안전장치 추가는 별도 이슈로 처리 예정.
    """
    tokenizer, model = load_artifacts()

    try:
        imap_server_input = input("사용할 메일서비스명 (gmail, naver): ").strip().lower()

        if imap_server_input == "gmail":
            imap_server = "imap.gmail.com"
            trash_folder = '"[Gmail]/Trash"'
        elif imap_server_input == "naver":
            imap_server = "imap.naver.com"
            trash_folder = "Trash"
        else:
            print("지원하지 않는 메일 서비스입니다.")
            return

        email_user = input("계정 ID: ")
        email_pass = input("password: ")

        if '@' not in email_user:
            if imap_server_input == "gmail":
                email_user += '@gmail.com'
            else:
                email_user += '@naver.com'

        important_count = 0
        spam_count = 0

        try:
            mail = imaplib.IMAP4_SSL(imap_server, 993)
            mail.login(email_user, email_pass)
            mail.timeout = 30
            print("로그인 성공!")

            if input("ㄱㄱ?: ") == 'ㄱㄱ':
                mail.select("inbox")

                search_condition = select_search_condition()
                try:
                    messages = search_with_retry(mail, search_condition)
                    message_ids = messages[0].split()

                    for index, msg_id in enumerate(message_ids, start=1):
                        print(f"{index}/{len(message_ids)} 번째 메일 처리 중...")

                        status, msg_data = mail.fetch(msg_id, "(RFC822)")
                        for response_part in msg_data:
                            if isinstance(response_part, tuple):
                                msg = email.message_from_bytes(response_part[1])
                                subject = safe_decode_header(msg["Subject"])

                                prediction = predict_spam(subject, tokenizer, model)

                                if prediction == 0:
                                    spam_count += 1
                                    if imap_server_input == "gmail":
                                        mail.store(msg_id, '+X-GM-LABELS', trash_folder)
                                    else:
                                        mail.store(msg_id, '+FLAGS', '\\Deleted')
                                    print(f"메일 {index}번을 휴지통으로 이동시킴.")
                                else:
                                    important_count += 1

                    print(f"중요 메일 {important_count}개 확인됨.")
                    print(f"스팸 메일 {spam_count}개 이동됨.")

                except Exception as e:
                    print(f"오류: {e}")

                if input("원상복구 ㄱㄱ?: ") == 'ㄱㄱ':
                    mail.select(trash_folder)
                    try:
                        messages = search_with_retry(mail, "ALL")
                        message_ids = messages[0].split()

                        for msg_id in message_ids:
                            if imap_server_input == "gmail":
                                mail.store(msg_id, '-X-GM-LABELS', trash_folder)
                                mail.store(msg_id, '+X-GM-LABELS', 'INBOX')
                            else:
                                mail.store(msg_id, '-FLAGS', '\\Deleted')
                                mail.store(msg_id, '+FLAGS', '\\Seen')

                            print(f"메일 {msg_id}을(를) 받은편지함으로 복구했습니다.")

                    except Exception as e:
                        print(f"검색 실패: {e}")

                mail.expunge()
                mail.close()
                mail.logout()

        except imaplib.IMAP4.error as e:
            print(f"IMAP 로그인 실패: {e}")

    except KeyboardInterrupt:
        print("\n프로그램이 사용자의 요청으로 종료되었습니다.")


if __name__ == "__main__":
    run_cleanup()
