package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"
	"unicode/utf8"

	"github.com/cloudwego/eino/schema"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/config"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/eval"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/llm"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

const testsetPath = "eval/testset.json"

// rulerPath 是"基准坐标系"：测试集里的 gold 块 ID 来自它（定长 500/50）。
// 换切块策略后块 ID 全变，靠基准索引把 gold 翻译成源文档字符区间，才能跨策略比较。
const rulerPath = ".data/idx-500-50.gob"

type strategy struct {
	name string
	path string
}

var strategies = []strategy{
	{"定长500/50", ".data/idx-500-50.gob"},
	{"定长500/0", ".data/idx-500-0.gob"},
	{"按标题800", ".data/idx-heading.gob"},
	{"按标题500", ".data/idx-heading500.gob"},
	{"按标题800无面包屑", ".data/idx-heading-nobc.gob"},
}

type acc struct {
	raw, adj [3]float64 // cov@5/10/20；adj 是扣掉"该策略不可达区域"后的校正值
}

type row struct {
	name    string
	chunks  int
	cov     map[string]*acc // 检索路线 → 累计值
	old     [4]float64      // 基准策略额外算：recall@5/10/20 + RR（2A 的老尺子）
	isRuler bool
}

func main() {
	only := flag.String("only", "", "只跑名字包含该子串的策略（如 500/0、heading）；空 = 全部")
	flag.Parse()

	ctx := context.Background()
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}
	embedder, err := llm.NewEmbedder(ctx, cfg)
	if err != nil {
		log.Fatal(err)
	}

	cases, err := eval.LoadTestset(testsetPath)
	if err != nil {
		log.Fatal(err)
	}

	ruler, err := rag.LoadIndex(rulerPath)
	if err != nil {
		log.Fatalf("加载基准索引失败（先构建）：go run ./cmd/index -strategy fixed -size 500 -overlap 50 -out %s\n%v", rulerPath, err)
	}
	if err := requireSpanMeta(ruler, "基准索引"); err != nil {
		log.Fatal(err)
	}
	goldSpans, goldSummary, err := buildGoldSpans(ruler, cases)
	if err != nil {
		log.Fatal(err)
	}

	docs, err := rag.LoadCorpus(ctx, "corpus")
	if err != nil {
		log.Fatal(err)
	}
	docLens := make(map[string]int, len(docs))
	for _, d := range docs {
		docLens[d.ID] = utf8.RuneCountInString(d.Content)
	}

	fmt.Printf("测试集：%s（%d 题）｜ gold 共 %s ｜ 基准坐标系：%s\n", testsetPath, len(cases), goldSummary, rulerPath)
	fmt.Printf("指标：coverage@k = 前 k 条命中的字符区间并集 ∩ gold 区间并集 的字符占比（与切块策略解耦）\n")
	fmt.Printf("      校正 = 扣掉该策略本身检索不到的区域之后的比例（按标题策略的标题行不在任何块的正文区间里）\n")
	fmt.Printf("候选池：每题 top-20（向量 / BM25 / RRF 三路各自算）\n\n")

	var rows []row
	for _, st := range strategies {
		if *only != "" && !strings.Contains(st.name, *only) {
			continue
		}
		idx, err := rag.LoadIndex(st.path)
		if err != nil {
			log.Printf("跳过 %s：加载 %s 失败（先构建：go run ./cmd/index -strategy ... -out %s）", st.name, st.path, st.path)
			continue
		}
		if err := requireSpanMeta(idx, st.name); err != nil {
			log.Fatal(err)
		}

		chunks := idx.ToDocuments()
		bm := rag.BuildBM25(chunks)
		vr := &rag.MemoryRetriever{Index: idx, Embedder: embedder, TopK: 20}
		isRuler := st.path == rulerPath

		r := row{name: st.name, chunks: len(chunks), isRuler: isRuler,
			cov: map[string]*acc{"向量": {}, "BM25": {}, "RRF": {}}}

		unreach, err := unreachableRegions(idx, docLens)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf("======== %s（%d 块）========\n", st.name, len(chunks))
		for i, c := range cases {
			vecDocs, err := vr.Retrieve(ctx, c.Question)
			if err != nil {
				log.Fatalf("%s 第 %d 题向量检索失败：%v", st.name, i+1, err)
			}
			bmDocs := bm.Search(c.Question, 20)
			fused := rag.FuseDocuments([][]*schema.Document{vecDocs, bmDocs}, 60)

			routes := []struct {
				name string
				docs []*schema.Document
			}{{"向量", vecDocs}, {"BM25", bmDocs}, {"RRF", fused}}

			goldUnion := eval.UnionLen(goldSpans[i])
			ceiling := goldUnion - eval.IntersectLen(goldSpans[i], unreach)
			goldLine := goldDesc(goldSpans[i])
			if ceiling < goldUnion {
				goldLine += fmt.Sprintf("｜本策略可达 %.0f%%", 100*float64(ceiling)/float64(goldUnion))
			}

			fmt.Printf("[%d] %s\n    gold：%s\n", i+1, c.Question, goldLine)
			for _, rt := range routes {
				spans, err := spansOf(rt.docs)
				if err != nil {
					log.Fatal(err)
				}
				vals := []float64{
					eval.CoverageAtK(spans, goldSpans[i], 5),
					eval.CoverageAtK(spans, goldSpans[i], 10),
					eval.CoverageAtK(spans, goldSpans[i], 20),
				}
				a := r.cov[rt.name]
				for j, v := range vals {
					a.raw[j] += v
					a.adj[j] += norm(v, goldUnion, ceiling)
				}
				fmt.Printf("      %-4s cov@5=%.2f  cov@10=%.2f  cov@20=%.2f\n", rt.name, vals[0], vals[1], vals[2])
			}
			if isRuler {
				ids := idsOf(vecDocs)
				r.old[0] += eval.RecallAtK(ids, c.Gold, 5)
				r.old[1] += eval.RecallAtK(ids, c.Gold, 10)
				r.old[2] += eval.RecallAtK(ids, c.Gold, 20)
				r.old[3] += eval.ReciprocalRank(ids, c.Gold)
			}
			fmt.Println()
		}
		rows = append(rows, r)
	}
	if len(rows) == 0 {
		log.Fatal("没有可跑的策略")
	}

	n := float64(len(cases))
	fmt.Println("──────── 汇总（平均）────────")
	fmt.Printf("%-14s %-5s cov@5   cov@10  cov@20 │ 校正@5  校正@10 校正@20\n", "策略", "路线")
	for _, r := range rows {
		for _, name := range []string{"向量", "BM25", "RRF"} {
			a := r.cov[name]
			fmt.Printf("%-14s %-5s %.3f   %.3f   %.3f   │ %.3f   %.3f   %.3f\n",
				fmt.Sprintf("%s(%d块)", r.name, r.chunks), name,
				a.raw[0]/n, a.raw[1]/n, a.raw[2]/n, a.adj[0]/n, a.adj[1]/n, a.adj[2]/n)
		}
	}
	for _, r := range rows {
		if r.isRuler {
			fmt.Printf("\n脚注：%s 用块 ID 老尺子复核（2A 口径，向量路）：recall@5=%.3f  recall@10=%.3f  recall@20=%.3f  MRR=%.3f\n",
				r.name, r.old[0]/n, r.old[1]/n, r.old[2]/n, r.old[3]/n)
		}
	}
	fmt.Printf("\n（每题每策略 1 次向量 API 调用，本次共 %d 次）\n", len(cases)*len(rows))
}

// requireSpanMeta 拦住 2B 之前建的旧索引：没有 start/end 就没法算覆盖率。
func requireSpanMeta(idx *rag.VectorIndex, name string) error {
	if len(idx.Entries) == 0 {
		return fmt.Errorf("%s 是空索引", name)
	}
	if _, ok := metaInt(idx.Entries[0].MetaData, "start"); !ok {
		return fmt.Errorf("%s 的块缺 start/end 元数据（2B 之前建的旧索引），请用 go run ./cmd/index 重建", name)
	}
	return nil
}

// buildGoldSpans 把测试集里的 gold 块 ID（基准坐标系）翻译成源文档字符区间。
func buildGoldSpans(ruler *rag.VectorIndex, cases []eval.TestCase) ([][]eval.Span, string, error) {
	byID := make(map[string]rag.IndexEntry, len(ruler.Entries))
	for _, e := range ruler.Entries {
		byID[e.ID] = e
	}
	out := make([][]eval.Span, len(cases))
	segs, total := 0, 0
	for i, c := range cases {
		for _, id := range c.Gold {
			e, ok := byID[id]
			if !ok {
				return nil, "", fmt.Errorf("第 %d 题 gold 块 %s 不在基准索引里（gold 与基准坐标系不同步）", i+1, id)
			}
			doc, _ := e.MetaData["doc"].(string)
			s, ok1 := metaInt(e.MetaData, "start")
			en, ok2 := metaInt(e.MetaData, "end")
			if doc == "" || !ok1 || !ok2 {
				return nil, "", fmt.Errorf("第 %d 题 gold 块 %s 缺 doc/start/end 元数据", i+1, id)
			}
			out[i] = append(out[i], eval.Span{Doc: doc, Start: s, End: en})
			segs++
		}
		total += eval.UnionLen(out[i])
	}
	return out, fmt.Sprintf("%d 段 / 并集 %d 字", segs, total), nil
}

// unreachableRegions 算出某策略在每篇文档里"任何块都覆盖不到"的字符区间：
// 块区间并集在 [0, 文档长度) 上的补集。定长策略铺满全文（空）；按标题策略会漏出每一行标题。
func unreachableRegions(idx *rag.VectorIndex, docLens map[string]int) ([]eval.Span, error) {
	byDoc := make(map[string][]eval.Span)
	for _, e := range idx.Entries {
		doc, _ := e.MetaData["doc"].(string)
		s, ok1 := metaInt(e.MetaData, "start")
		en, ok2 := metaInt(e.MetaData, "end")
		if doc == "" || !ok1 || !ok2 {
			return nil, fmt.Errorf("块 %s 缺 span 元数据（索引要重建）", e.ID)
		}
		byDoc[doc] = append(byDoc[doc], eval.Span{Doc: doc, Start: s, End: en})
	}
	var out []eval.Span
	for doc, spans := range byDoc {
		n, ok := docLens[doc]
		if !ok {
			return nil, fmt.Errorf("corpus 里没有 %s（索引与语料不一致？）", doc)
		}
		pos := 0
		for _, m := range eval.MergeSpans(spans) {
			if m.Start > pos {
				out = append(out, eval.Span{Doc: doc, Start: pos, End: m.Start})
			}
			pos = max(pos, m.End)
		}
		if pos < n {
			out = append(out, eval.Span{Doc: doc, Start: pos, End: n})
		}
	}
	return out, nil
}

// norm 把原始覆盖率换算成"扣掉不可达区域"后的比例：cov × 并集 / 可达。
func norm(cov float64, union, ceiling int) float64 {
	if ceiling <= 0 {
		return 0
	}
	return cov * float64(union) / float64(ceiling)
}

// spansOf 按相关性顺序取出每条检索结果的字符区间。
func spansOf(docs []*schema.Document) ([]eval.Span, error) {
	out := make([]eval.Span, 0, len(docs))
	for _, d := range docs {
		doc, _ := d.MetaData["doc"].(string)
		s, ok1 := metaInt(d.MetaData, "start")
		en, ok2 := metaInt(d.MetaData, "end")
		if doc == "" || !ok1 || !ok2 {
			return nil, fmt.Errorf("块 %s 缺 span 元数据（索引要重建）", d.ID)
		}
		out = append(out, eval.Span{Doc: doc, Start: s, End: en})
	}
	return out, nil
}

func metaInt(m map[string]any, key string) (int, bool) {
	switch v := m[key].(type) {
	case int:
		return v, true
	case int64:
		return int(v), true
	case float64:
		return int(v), true
	}
	return 0, false
}

func idsOf(docs []*schema.Document) []string {
	out := make([]string, len(docs))
	for i, d := range docs {
		out[i] = d.ID
	}
	return out
}

func goldDesc(spans []eval.Span) string {
	var b strings.Builder
	for i, s := range spans {
		if i > 0 {
			b.WriteString("、")
		}
		fmt.Fprintf(&b, "%s[%d,%d)", s.Doc, s.Start, s.End)
	}
	fmt.Fprintf(&b, "（%d 段 / 并集 %d 字）", len(spans), eval.UnionLen(spans))
	return b.String()
}
