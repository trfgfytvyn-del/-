# filename: app.py
import streamlit as st
import random
import pandas as pd
from collections import defaultdict
from itertools import combinations

st.set_page_config(page_title="로테이션 소개팅 자리 배치기 (성별 구성 지정)", layout="wide")

# -----------------------------
# 파서 & 유틸
# -----------------------------
def parse_names(text: str):
    if not text.strip():
        return []
    raw = [t.strip() for chunk in text.split("\n") for t in chunk.replace(",", "\n").split("\n")]
    return [x for x in raw if x]

def parse_soft_avoid(text: str):
    """
    형식:
    정수진: 김강모, 이지선
    김강모: 정수진
    (단방향 패널티)
    """
    m = defaultdict(set)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines:
        if ":" in ln:
            left, right = ln.split(":", 1)
            person = left.strip()
            others = [x.strip() for x in right.replace("、", ",").split(",") if x.strip()]
            for o in others:
                m[person].add(o)
    return m

def parse_table_blueprints(text: str):
    """
    한 줄에 한 테이블
    예시:
      3:M1,F2
      4:M2,F2
    반환: [{"size":3,"M":1,"F":2}, ...]
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    blueprints = []
    for ln in lines:
        if ":" not in ln:
            return None, f"'{ln}' -> 콜론(:)이 없습니다. 예: 3:M1,F2"
        left, right = ln.split(":", 1)
        if not left.isdigit():
            return None, f"'{ln}' -> 앞부분은 전체 인원 수(정수)여야 합니다. 예: 3:M1,F2"
        size = int(left)
        # 오른쪽에서 Mx, Fy 추출
        parts = [p.strip() for p in right.split(",") if p.strip()]
        M = F = None
        for p in parts:
            p = p.upper()
            if p.startswith("M"):
                try:
                    M = int(p[1:])
                except:
                    return None, f"'{ln}' -> M 다음엔 숫자가 와야 합니다. 예: M1"
            elif p.startswith("F"):
                try:
                    F = int(p[1:])
                except:
                    return None, f"'{ln}' -> F 다음엔 숫자가 와야 합니다. 예: F2"
            else:
                return None, f"'{ln}' -> 'M숫자,F숫자' 형식이어야 합니다. 예: M1,F2"
        if M is None or F is None:
            return None, f"'{ln}' -> M과 F 모두 지정해야 합니다. 예: 3:M1,F2"
        if M + F != size:
            return None, f"'{ln}' -> M({M})+F({F}) != 총인원({size})"
        blueprints.append({"size": size, "M": M, "F": F})
    return blueprints, ""

def pairs_in_table(table):
    return [frozenset({a, b}) for a, b in combinations(table, 2)]

def soft_penalty_table(table, soft_avoid):
    # 단방향 패널티 합산
    penalty = 0
    members = set(table)
    for a in table:
        avoid_set = soft_avoid.get(a, set())
        penalty += len(members.intersection(avoid_set))
    return penalty

# -----------------------------
# 핵심 로직 (백트래킹)
# -----------------------------
def build_one_round(men, women, blueprints, meet_history, soft_avoid, seed=None):
    """
    하드 제약:
      - 과거 만난 쌍 금지
      - 각 테이블의 남/녀 수를 blueprint에 맞게 정확히 충족
    소프트 제약:
      - soft_avoid 패널티 최소화 (가능하면 회피, 불가피 시 경고)
    """
    if seed is not None:
        random.seed(seed)

    total_people = len(men) + len(women)
    if sum(bp["size"] for bp in blueprints) != total_people:
        return None, None, "테이블 총 인원 합이 전체 인원과 다릅니다."

    if sum(bp["M"] for bp in blueprints) != len(men) or sum(bp["F"] for bp in blueprints) != len(women):
        return None, None, "테이블별 M/F 합계가 실제 남/여 인원 수와 일치해야 합니다."

    # 탐색 대상 풀
    men_pool = men[:]
    women_pool = women[:]
    random.shuffle(men_pool)
    random.shuffle(women_pool)

    # 테이블별 현재 좌석
    tables = [[] for _ in blueprints]
    # 남/여 남은 슬롯
    slots = [{"M": bp["M"], "F": bp["F"]} for bp in blueprints]

    best_solution = None
    best_penalty = 10**9

    # 후보 선택 휴리스틱: soft_avoid가 많은(제약 높은) 사람 먼저
    def next_person():
        remaining = men_pool + women_pool
        if not remaining:
            return None
        remaining.sort(key=lambda x: len(soft_avoid.get(x, set())), reverse=True)
        return remaining[0]

    def remove_person(p):
        if p in men_pool:
            men_pool.remove(p)
            return "M"
        else:
            women_pool.remove(p)
            return "F"

    def add_person_back(p, gender):
        if gender == "M":
            men_pool.append(p)
        else:
            women_pool.append(p)

    def can_add(table_idx, person):
        for other in tables[table_idx]:
            if frozenset({person, other}) in meet_history:
                return False
        return True

    def backtrack():
        nonlocal best_solution, best_penalty

        p = next_person()
        if p is None:
            penalty = sum(soft_penalty_table(t, soft_avoid) for t in tables)
            if penalty < best_penalty:
                best_penalty = penalty
                best_solution = [t[:] for t in tables]
            return

        gender = remove_person(p)

        candidates = []
        for idx, (t, sl) in enumerate(zip(tables, slots)):
            if sl[gender] <= 0:
                continue
            if not can_add(idx, p):
                continue
            before = soft_penalty_table(t, soft_avoid)
            t.append(p)
            after = soft_penalty_table(t, soft_avoid)
            delta = after - before
            t.pop()
            candidates.append((delta, idx))

        candidates.sort(key=lambda x: x[0])

        for _, idx in candidates:
            tables[idx].append(p)
            slots[idx][gender] -= 1
            backtrack()
            slots[idx][gender] += 1
            tables[idx].pop()

        add_person_back(p, gender)

    backtrack()

    if best_solution is None:
        return None, None, "해당 성별 구성/재만남 금지 조건으로는 배치가 불가능합니다. 구성 또는 인원을 조정해 주세요."

    warnings = []
    for ti, t in enumerate(best_solution):
        for a in t:
            for b in t:
                if a == b:
                    continue
                if b in soft_avoid.get(a, set()):
                    warnings.append(f"[테이블 {ti+1}] {a} ↔ {b} (희망 회피였으나 불가피하게 함께 배치)")

    return best_solution, best_penalty, "\n".join(sorted(set(warnings))) if warnings else ""

def round_to_dataframe(round_tables, title="Round"):
    max_len = max(len(t) for t in round_tables) if round_tables else 0
    data = {}
    for i, t in enumerate(round_tables, start=1):
        col = [f"Table {i}"] + t + [""]*(max_len - len(t))
        data[f"Table {i}"] = col
    df = pd.DataFrame(data)
    df.index = [""] + [f"Seat {i}" for i in range(1, max_len+1)]
    df.index.name = title
    return df

# -----------------------------
# UI
# -----------------------------
st.title("🪑 로테이션 소개팅 자리 배치 (테이블별 성별 구성 지정)")
st.caption("하드 제약: (1) 이전 라운드에 만난 사람은 다시 같은 테이블 금지 (2) 테이블별 M/F 구성 정확히 충족\n소프트 제약: 피하고 싶은 조합은 가능하면 회피(불가피 시 경고 표시)")

colA, colB = st.columns(2)
with colA:
    men_text = st.text_area("남자 이름 (줄바꿈/쉼표 구분)", height=160, placeholder="예) 전준형, 오승인, 김찬우\n...")
with colB:
    women_text = st.text_area("여자 이름 (줄바꿈/쉼표 구분)", height=160, placeholder="예) 정수진, 최다연, 박가예\n...")

st.markdown("**테이블 구성 (한 줄 = 한 테이블)**  \n형식: `총인원:M숫자,F숫자` (예: `3:M1,F2` / `4:M2,F2`)")
table_bp_text = st.text_area(
    "테이블별 성별 구성",
    height=150,
    placeholder="3:M1,F2\n4:M2,F2\n4:M3,F1"
)

rounds = st.number_input("라운드 수", min_value=1, max_value=10, value=3, step=1)

st.markdown("**소프트 제약(피하고 싶은 조합) — 최하우선, 불가피 시 경고**")
soft_avoid_text = st.text_area(
    "형식: 이름: 상대1, 상대2",
    height=120,
    placeholder="정수진: 김강모, 이지선\n김강모: 정수진"
)

if st.button("자리 배치 생성"):
    men = parse_names(men_text)
    women = parse_names(women_text)

    blueprints, err = parse_table_blueprints(table_bp_text)
    if err:
        st.error(err)
        st.stop()
    if not men and not women:
        st.error("참가자 이름을 입력해 주세요.")
        st.stop()
    if sum(bp["M"] for bp in blueprints) != len(men):
        st.error(f"테이블의 M 총합({sum(bp['M'] for bp in blueprints)})이 실제 남자 인원({len(men)})과 같아야 합니다.")
        st.stop()
    if sum(bp["F"] for bp in blueprints) != len(women):
        st.error(f"테이블의 F 총합({sum(bp['F'] for bp in blueprints)})이 실제 여자 인원({len(women)})과 같아야 합니다.")
        st.stop()

    soft_avoid = parse_soft_avoid(soft_avoid_text)

    meet_history = set()
    all_rounds = []
    all_warnings = []

    for r in range(1, rounds+1):
        solution, penalty, warn = build_one_round(
            men, women, blueprints, meet_history, soft_avoid, seed=777 + r
        )

        st.subheader(f"Round {r}")
        if solution is None:
            st.error(penalty if isinstance(penalty, str) else "해답을 찾지 못했습니다.")
            break

        df = round_to_dataframe(solution, title=f"Round {r}")
        st.dataframe(df, use_container_width=True)

        if warn:
            st.warning("소프트 제약(피하고 싶은 사람) 위반 발생:\n" + warn)
            all_warnings.append((r, warn))

        # 하드 제약 업데이트
        for table in solution:
            for p in combinations(table, 2):
                meet_history.add(frozenset(p))

        all_rounds.append(solution)

    if all_rounds:
        # CSV 내보내기
        rows = []
        for ri, tables in enumerate(all_rounds, start=1):
            for ti, t in enumerate(tables, start=1):
                for si, name in enumerate(t, start=1):
                    rows.append({"Round": ri, "Table": ti, "Seat": si, "Name": name})
        out = pd.DataFrame(rows)
        st.download_button(
            "📥 전체 배치 CSV 다운로드",
            data=out.to_csv(index=False).encode("utf-8-sig"),
            file_name="rotation_seating_gender_blueprint.csv",
            mime="text/csv"
        )

        if all_warnings:
            st.info("라운드별 소프트 제약 위반 요약")
            for r, w in all_warnings:
                with st.expander(f"Round {r} 경고 보기"):
                    st.write(w)
