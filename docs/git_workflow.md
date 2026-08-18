# Git 작업 방식

버전 관리를 일관되게 하기 위한 규칙입니다.

## 브랜치 구조

| 브랜치 | 역할 |
|---|---|
| `main` | **안정 버전.** 여러 번 검증되어 안정적임이 확실해진 코드만 존재. |
| `develop` | **핵심 개발 라인.** 기능 브랜치가 여기로 모임. |
| `feat/*`, `fix/*`, `docs/*`, `exp/*`, `refactor/*` | **작업 브랜치.** 개별 기능/수정/실험 단위. |

## 작업 흐름

```
feat/xxx  ──(개발·커밋)──▶  develop  ──(안정성 확인)──▶  main
                  ▲                 ▲                          ▲
             새 작업마다        사용자가 직접 merge       사용자가 직접 merge
             브랜치 생성
```

1. 새 작업을 시작할 때마다 `develop`에서 **작업 브랜치를 새로 생성**한다.
2. 해당 브랜치에서 기능을 개발하고 **commit**한다.
3. 사용자 환경에서 **오류 없이 잘 동작함을 확인**한다.
4. 확인이 끝나면 **사용자가 직접** `develop`로 merge한다.
5. `develop`에서 여러 번 운용해 **안정성이 확실해지면**, **사용자가 직접** `main`으로 merge한다.

## ⚠️ 핵심 규칙 (AI agent/자동화 대상)

- **`commit`은 자율적으로 수행해도 된다.**
- **`merge`와 `push`는 절대 임의로 수행하지 않는다. 반드시 사용자가 직접 수행한다.**
- 작업은 항상 작업 브랜치에서 진행하고, `develop`/`main`에 직접 커밋하지 않는다.

## 커밋 이력을 다시 쓸 때 (2026-08-18 실측 규칙)

**문서가 커밋 해시를 실측의 출처로 참조한다.** 예: "photometric augmentation (런 5, commit
`709ff36`)", "3-class 역빈도 가중은 커밋 `dd3d474`으로 추가했다". 어떤 숫자가 어느 코드 상태에서
나왔는지를 추적하는 장치이므로, 이력을 다시 쓰면 **참조를 함께 갱신해야 한다.**

- rewrite 전에 **`grep -rnoE '\b[0-9a-f]{7}\b' docs/ AGENTS.md`로 참조를 먼저 찾는다.**
  단 `develop`/`main`에 이미 있는 커밋은 rewrite 대상이 아니므로
  `git merge-base --is-ancestor <sha> origin/develop`로 걸러낸다.
- **백업 태그를 먼저 박는다**: `git tag pre-consolidate-backup`. 되돌리기가 한 줄이 된다.
- **인접한 커밋만 합치고 순서는 바꾸지 않는다.** 인접 squash는 충돌이 원리적으로 나지 않는다.
  리네임을 가로질러 reorder하면 충돌이 난다.
- 검증은 **트리 동일성**으로 한다: `git diff <백업태그> HEAD`가 비어야 한다(문서 갱신분 제외).
  그리고 테스트를 다시 돌린다.
- **`--squash` merge는 쓰지 않는다.** 브랜치를 지우면 커밋이 사라져 위 참조가 전부 깨진다
  (실측 확인함). 일반 merge면 커밋이 develop의 조상이 되어 브랜치를 지워도 살아 있다.

2026-08-18에 `feat/bev-free-space-task`의 69개 커밋을 18개로 통합했다. 그 69개는
`type: subject` 규칙을 **0/69** 지켰고, 21개가 진행상황 원장이나 리뷰 후속 수정이었다.

## 브랜치 네이밍 컨벤션

- `feat/<간단한-설명>` — 새 기능 (예: `feat/woodscape-loader`)
- `fix/<간단한-설명>` — 버그 수정
- `docs/<간단한-설명>` — 문서 작업
- `exp/<간단한-설명>` — 실험적 시도
- `refactor/<간단한-설명>` — 리팩터링

## 커밋 단위 & 메시지 컨벤션

- **커밋은 큰 논리 단위로 남긴다.** 자잘한 변경 하나하나를 개별 커밋으로 쪼개지 말고, 의미 있는 작업 단위(예: "데이터 로더 추가", "문서 초기화")로 묶는다.
- **커밋 메시지는 영어로 작성한다.** (파일/문서 내용은 한글도 무방하지만 커밋 메시지는 영어)
- 형식: `type: subject`
- type 예시: `feat`, `fix`, `docs`, `refactor`, `chore`, `exp`
- 예: `feat: add initial WoodScape data loader`, `docs: expand roadmap phase 2`

## 자주 쓰는 명령어

```bash
# 작업 브랜치 생성 & 이동
git switch develop
git switch -c feat/woodscape-loader

# 커밋
git add <files>
git commit -m "feat: ..."

# (사용자가 직접 수행) develop로 merge
git switch develop
git merge --no-ff feat/woodscape-loader

# (사용자가 직접 수행) 원격 반영
git push origin develop
```
