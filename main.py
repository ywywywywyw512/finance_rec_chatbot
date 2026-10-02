import os
import hashlib
import secrets
import shutil
from pathlib import Path

import pandas as pd
import uvicorn

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from litedb import DiskDatabase
from openai import OpenAI


# ==========================================================
# 기본 경로 설정
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
PROFILE_DIR = STATIC_DIR / "profiles"
DB_DIR = BASE_DIR / "litedb_data"

CSV_PATH = STATIC_DIR / "예금목록.csv"

STATIC_DIR.mkdir(exist_ok=True)
PROFILE_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)


# ==========================================================
# ENV
# ==========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(
    title="Finanfit"
)

app.add_middleware(
    SessionMiddleware,
    secret_key="finanfit-2026-secret-key"
)

app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ),
    name="static"
)

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# ==========================================================
# LiteDB
# ==========================================================

db = DiskDatabase(
    str(DB_DIR)
)


class User:

    def __init__(
        self,
        username="",
        password="",
        name="",
        email="",
        age=0,
        gender="",
        investment_type="",
        job="",
        income=0,
        profile_image=""
    ):

        self.username = username
        self.password = password
        self.name = name
        self.email = email
        self.age = age
        self.gender = gender
        self.investment_type = investment_type
        self.job = job
        self.income = income
        self.profile_image = profile_image

    def __repr__(self):
        return self.username


# ==========================================================
# 사용자 관련 함수
# ==========================================================

def get_user(username):

    try:

        users = list(
            db.select(User).retrieve(
                username=username
            )
        )

        if users:
            return users[0]

    except Exception:
        pass

    return None


def hash_password(password):

    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    ).hex()

    return f"{salt}${password_hash}"


def verify_password(
    password,
    saved_password
):

    try:

        salt, saved_hash = saved_password.split(
            "$"
        )

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000
        ).hex()

        return secrets.compare_digest(
            password_hash,
            saved_hash
        )

    except Exception:

        return False


def current_user(request):

    username = request.session.get(
        "username"
    )

    if not username:
        return None

    return get_user(username)


# ==========================================================
# CSV 읽기
# ==========================================================

def load_products():

    if not CSV_PATH.exists():
        return pd.DataFrame()

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr"
    ]:

        try:

            df = pd.read_csv(
                CSV_PATH,
                encoding=encoding
            )

            return df.fillna("")

        except Exception:
            pass

    return pd.DataFrame()


# ==========================================================
# 상품 추천 알고리즘
# ==========================================================

def recommend_products(
    user,
    count=3
):

    df = load_products()

    if df.empty:
        return []

    # ----------------------------------------------
    # 금리 숫자로 변환
    # ----------------------------------------------

    if "최고금리_숫자" in df.columns:

        df["rate"] = pd.to_numeric(
            df["최고금리_숫자"],
            errors="coerce"
        ).fillna(0)

    else:

        df["rate"] = pd.to_numeric(
            df["최고금리"],
            errors="coerce"
        ).fillna(0)


    # 추천 점수
    df["score"] = 0.0


    # ======================================================
    # 1. 금리
    # ======================================================

    max_rate = df["rate"].max()

    if max_rate > 0:

        df["score"] += (
            df["rate"] / max_rate
        ) * 50


    # ======================================================
    # 2. 투자성향
    # ======================================================

    if user.investment_type == "안정형":

        df["score"] += 25

    elif user.investment_type == "안정추구형":

        df["score"] += 20

    elif user.investment_type == "위험중립형":

        df["score"] += 15

    elif user.investment_type == "적극투자형":

        df["score"] += 10

    else:

        df["score"] += 5


    # ======================================================
    # 3. 나이
    # ======================================================

    if int(user.age) <= 29:

        keywords = "청년|첫거래|직장인|첫만남"

        if "상세정보전체" in df.columns:

            condition = (
                df["상세정보전체"]
                .astype(str)
                .str.contains(
                    keywords,
                    case=False,
                    na=False
                )
            )

            df.loc[
                condition,
                "score"
            ] += 15


    elif int(user.age) >= 50:

        keywords = "연금|노후|시니어"

        if "상세정보전체" in df.columns:

            condition = (
                df["상세정보전체"]
                .astype(str)
                .str.contains(
                    keywords,
                    case=False,
                    na=False
                )
            )

            df.loc[
                condition,
                "score"
            ] += 15


    # ======================================================
    # 4. 직업
    # ======================================================

    if user.job in [
        "직장인",
        "공무원",
        "공기업"
    ]:

        if "상세정보전체" in df.columns:

            condition = (
                df["상세정보전체"]
                .astype(str)
                .str.contains(
                    "급여|직장|주거래",
                    case=False,
                    na=False
                )
            )

            df.loc[
                condition,
                "score"
            ] += 10


    # ======================================================
    # 5. 소득
    # ======================================================

    try:
        income = int(user.income)

    except Exception:
        income = 0


    if income <= 3000:

        if "가입금액" in df.columns:

            condition = (
                df["가입금액"]
                .astype(str)
                .str.contains(
                    "1만원|10만원|100만원",
                    na=False
                )
            )

            df.loc[
                condition,
                "score"
            ] += 8


    # ======================================================
    # 추천 순위
    # ======================================================

    recommended = (
        df.sort_values(
            by=[
                "score",
                "rate"
            ],
            ascending=False
        )
        .head(count)
    )


    result = []

    for _, row in recommended.iterrows():

        result.append({

            "bank":
                str(
                    row.get(
                        "금융사",
                        "금융사 없음"
                    )
                ),

            "name":
                str(
                    row.get(
                        "상품명",
                        "상품명 없음"
                    )
                ),

            "max_rate":
                str(
                    row.get(
                        "최고금리",
                        "-"
                    )
                ),

            "base_rate":
                str(
                    row.get(
                        "기본금리",
                        "-"
                    )
                ),

            "period":
                str(
                    row.get(
                        "가입기간",
                        "-"
                    )
                ),

            "amount":
                str(
                    row.get(
                        "가입금액",
                        "-"
                    )
                ),

            "method":
                str(
                    row.get(
                        "가입방법",
                        "-"
                    )
                ),

            "target":
                str(
                    row.get(
                        "가입대상",
                        "-"
                    )
                ),

            "url":
                str(
                    row.get(
                        "상세URL",
                        ""
                    )
                ),

            "score":
                round(
                    float(
                        row["score"]
                    ),
                    1
                )
        })

    return result


# ==========================================================
# 로그인 페이지
# ==========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def login_page(
    request: Request
):

    if current_user(request):

        return RedirectResponse(
            "/main",
            status_code=302
        )

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": None
        }
    )


# ==========================================================
# 로그인
# ==========================================================

@app.post(
    "/login",
    response_class=HTMLResponse
)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):

    user = get_user(
        username
    )

    if not user:

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error":
                    "아이디 또는 비밀번호를 확인해주세요."
            }
        )

    if not verify_password(
        password,
        user.password
    ):

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error":
                    "아이디 또는 비밀번호를 확인해주세요."
            }
        )


    request.session[
        "username"
    ] = username


    return RedirectResponse(
        "/main",
        status_code=302
    )


# ==========================================================
# 회원가입 페이지
# ==========================================================

@app.get(
    "/signup",
    response_class=HTMLResponse
)
async def signup_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={
            "error": None
        }
    )


# ==========================================================
# 회원가입
# ==========================================================

@app.post(
    "/signup",
    response_class=HTMLResponse
)
async def signup(
    request: Request,

    username: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),

    age: int = Form(...),
    gender: str = Form(...),

    investment_type: str = Form(...),

    job: str = Form(...),

    income: int = Form(...),

    profile_image: UploadFile = File(None)
):


    # 아이디 중복
    if get_user(username):

        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={
                "error":
                    "이미 존재하는 아이디입니다."
            }
        )


    image_path = ""


    # 프로필 이미지
    if (
        profile_image
        and
        profile_image.filename
    ):

        extension = Path(
            profile_image.filename
        ).suffix

        filename = (
            username
            +
            "_profile"
            +
            extension
        )

        save_path = (
            PROFILE_DIR
            /
            filename
        )

        with open(
            save_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                profile_image.file,
                buffer
            )

        image_path = (
            "/static/profiles/"
            +
            filename
        )


    user = User(

        username=username,

        password=hash_password(
            password
        ),

        name=name,

        email=email,

        age=age,

        gender=gender,

        investment_type=investment_type,

        job=job,

        income=income,

        profile_image=image_path
    )


    try:

        db.select(
            User
        ).insert(
            user
        )

    except Exception:

        try:
            db.insert(user)

        except Exception as e:

            print(
                "LiteDB 저장 오류:",
                e
            )

            return templates.TemplateResponse(
                request=request,
                name="signup.html",
                context={
                    "error":
                        "회원가입 중 DB 오류가 발생했습니다."
                }
            )


    request.session[
        "username"
    ] = username


    return RedirectResponse(
        "/main",
        status_code=302
    )


# ==========================================================
# 메인페이지
# ==========================================================

@app.get(
    "/main",
    response_class=HTMLResponse
)
async def main_page(
    request: Request
):

    user = current_user(
        request
    )

    if not user:

        return RedirectResponse(
            "/",
            status_code=302
        )


    products = recommend_products(
        user,
        3
    )


    return templates.TemplateResponse(
        request=request,
        name="main.html",
        context={

            "user":
                user,

            "products":
                products

        }
    )


# ==========================================================
# 챗봇 화면
# ==========================================================

@app.get(
    "/chatbot",
    response_class=HTMLResponse
)
async def chatbot_page(
    request: Request
):

    user = current_user(
        request
    )

    if not user:

        return RedirectResponse(
            "/",
            status_code=302
        )


    return templates.TemplateResponse(
        request=request,
        name="chatbot.html",
        context={
            "user": user
        }
    )


# ==========================================================
# 챗봇 API
# ==========================================================

@app.post(
    "/api/chat"
)
async def chat_api(
    request: Request
):

    user = current_user(
        request
    )

    if not user:

        return JSONResponse(
            {
                "answer":
                    "로그인이 필요합니다."
            },
            status_code=401
        )


    body = await request.json()

    message = body.get(
        "message",
        ""
    )


    if not message.strip():

        return {
            "answer":
                "궁금한 금융상품을 입력해주세요."
        }


    df = load_products()


    # 챗봇에게 CSV 데이터 일부 전달
    product_text = ""


    if not df.empty:

        columns = [
            "금융사",
            "상품명",
            "최고금리",
            "기본금리",
            "가입기간",
            "가입금액",
            "가입방법",
            "가입대상"
        ]


        columns = [
            col
            for col in columns
            if col in df.columns
        ]


        # 금리 높은 상품 20개
        if "최고금리_숫자" in df.columns:

            chatbot_df = (
                df.sort_values(
                    "최고금리_숫자",
                    ascending=False
                )
                .head(20)
            )

        else:

            chatbot_df = df.head(
                20
            )


        product_text = (
            chatbot_df[
                columns
            ]
            .to_string(
                index=False
            )
        )


    system_prompt = f"""
당신은 Finanfit의 금융상품 상담 AI입니다.

사용자의 정보는 다음과 같습니다.

이름: {user.name}
나이: {user.age}
성별: {user.gender}
투자성향: {user.investment_type}
직업: {user.job}
연소득: {user.income}만원

아래는 예금목록.csv에 실제로 존재하는 금융상품 데이터 일부입니다.

--------------------
{product_text}
--------------------

규칙:

1. CSV에 존재하는 상품을 우선적으로 설명합니다.
2. 없는 상품을 실제 상품처럼 만들어내지 마세요.
3. 사용자의 나이, 투자성향, 직업, 소득을 고려해서 설명하세요.
4. 최고금리와 기본금리가 다르면 반드시 구분해서 설명하세요.
5. 금융상품 가입 전 실제 금융회사에서 최신 조건을 확인해야 한다고 안내하세요.
6. 너무 딱딱하지 않은 친절한 금융상담 말투를 사용하세요.
7. 투자 성과나 수익을 보장하지 마세요.
"""


    try:

        response = client.responses.create(

            model="gpt-5-mini",

            instructions=system_prompt,

            input=message
        )


        answer = response.output_text


    except Exception as e:

        print(
            "OpenAI API 오류:",
            e
        )

        answer = (
            "현재 AI 상담 연결 중 오류가 발생했습니다. "
            "API Key와 인터넷 연결을 확인해주세요."
        )


    return {
        "answer":
            answer
    }


# ==========================================================
# 로그아웃
# ==========================================================

@app.get(
    "/logout"
)
async def logout(
    request: Request
):

    request.session.clear()

    return RedirectResponse(
        "/",
        status_code=302
    )


# ==========================================================
# 실행
# ==========================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=1439,
        reload=True
    )
    