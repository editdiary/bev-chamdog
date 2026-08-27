# docs/figures

논문·발표용 그림. **그림의 정본은 생성 스크립트다** -- SVG를 직접 편집하면 다음 재생성에서 사라진다.

| 파일 | 무엇 | 어디에 쓰나 |
|---|---|---|
| `make_pipeline_figure.py` | 파이프라인 모식도 **생성기 (정본)** | 배치·좌표를 고칠 때 |
| `bev_pipeline.svg` | 독립 SVG (색 리터럴, 폰트 폴백) | **논문 figure**, Word, Inkscape |
| `bev_pipeline_themed.svg` | 색이 CSS 변수인 판본 | `bev_pipeline.html`에 인라인됨 |
| `bev_pipeline.html` | 모식도 + 형상표 + 설명 한 페이지 | 브라우저로 읽기, Ctrl+P로 PDF 저장 |

재생성:

```bash
python docs/figures/make_pipeline_figure.py
```

`bev_pipeline.html`은 손으로 쓴 파일이고 SVG만 위 스크립트에서 온다. 스크립트를 고쳤으면
`bev_pipeline_themed.svg`의 내용을 HTML의 `<svg ...>...</svg>` 자리에 다시 붙여야 한다.

## 논문에 넣기

LaTeX는 SVG를 직접 못 읽으므로 PDF로 바꾼다(둘 중 편한 쪽):

```bash
rsvg-convert -f pdf -o bev_pipeline.pdf docs/figures/bev_pipeline.svg
inkscape docs/figures/bev_pipeline.svg --export-type=pdf
```

한글 라벨이 있으므로 변환 환경에 한글 폰트(예: Noto Sans KR)가 있어야 한다. 없으면
`make_pipeline_figure.py`의 라벨 문자열을 영문으로 바꿔 다시 뽑는 편이 빠르다.

## 그림에 적힌 숫자의 출처

형상·파라미터 수는 저장소 코드를 실행해 확인한 값이고 `docs/paper_experiment_compendium.md`
§1.3과 일치한다(총 40,999,746 = 41.00 M). 디코더 수용영역 118셀은 정규화·활성을 선형화한
뒤 출력 한 칸의 gradient가 닿는 입력 범위를 측정한 값이다.
