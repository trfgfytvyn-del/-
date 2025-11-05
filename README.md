
# 로테이션 소개팅 자리 배치기 (배포용)

Streamlit 기반의 웹 앱입니다. 테이블별 **정확한 M/F 구성**을 지정하고, **여러 라운드**에 걸쳐 **이전에 만난 사람은 다시 만나지 않도록** 배치합니다.  
또한 "피하고 싶은 사람"은 **소프트 제약**으로 최대한 회피하며, 불가피할 경우 결과에 **경고**가 표시됩니다.

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud에 배포
1. 이 폴더를 GitHub 저장소로 올립니다.
2. https://streamlit.io/cloud 에서 **New app** → 저장소 선택 → `app.py` 지정 → Deploy
3. 배포되면 `https://<username>-<repo>-streamlit.app` 형태의 URL이 생성됩니다.

## Hugging Face Spaces에 배포
1. https://huggingface.co/spaces 에서 **Create new Space**
2. Space SDK는 **Streamlit** 선택
3. 이 폴더의 파일(`app.py`, `requirements.txt`) 업로드
4. 자동 빌드 후 생성되는 Space URL로 접속
