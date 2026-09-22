package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/cloudwego/eino/schema"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/config"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/eval"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/llm"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

const testsetPath = "eval/testset.json"

// kLadder 覆盖率报告的 k 阶梯：@1 只看头名块（天然排名敏感，取代单独的 MRR 一列），
// @20 看深池（候选池上限），中间三档观察"多带回一点"的边际收益。
var kLadder = [5]int{1, 3, 5, 10, 20}

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
	raw, adj [5]float64 // 与 kLadder 对齐；adj 是扣掉"该策略不可达区域"后的校正值
}

type row struct {
	name   string
	chunks int
	cov    map[string]*acc // 检索路线 → 累计值
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

	docs, err := rag.LoadCorpus(ctx, "corpus")
	if err != nil {
		log.Fatal(err)
	}
	docLens := make(map[string]int, len(docs))
	for _, d := range docs {
		docLens[d.ID] = utf8.RuneCountInString(d.Content)
	}
	if err := verifyGold(cases, docs); err != nil {
		log.Fatal(err)
	}
	printHeader(cases)

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

		r := row{name: st.name, chunks: len(chunks),
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

			gold := c.Spans()
			goldUnion := eval.UnionLen(gold)
			ceiling := goldUnion - eval.IntersectLen(gold, unreach)
			goldLine := goldDesc(c)
			if ceiling < goldUnion {
				goldLine += fmt.Sprintf("｜本策略可达 %.0f%%", 100*float64(ceiling)/float64(goldUnion))
			}

			fmt.Printf("[%d] %s  [%s]\n    gold：%s\n", i+1, c.Question, strings.Join(c.Tags, "/"), goldLine)
			for _, rt := range routes {
				spans, err := spansOf(rt.docs)
				if err != nil {
					log.Fatal(err)
				}
				vals := make([]float64, len(kLadder))
				for j, k := range kLadder {
					vals[j] = eval.CoverageAtK(spans, gold, k)
				}
				a := r.cov[rt.name]
				for j, v := range vals {
					a.raw[j] += v
					a.adj[j] += norm(v, goldUnion, ceiling)
				}
				fmt.Printf("      %-4s %s\n", rt.name, ladder(vals))
			}
			fmt.Println()
		}
		rows = append(rows, r)
	}
	if len(rows) == 0 {
		log.Fatal("没有可跑的策略")
	}

	n := float64(len(cases))
	fmt.Println("──────── 汇总（每题先算值，再按题 macro 平均）────────")
	printTable("覆盖率 cov@k（原始）", rows, n, func(a *acc) [5]float64 { return a.raw })
	printTable("覆盖率 校正@k（扣掉该策略不可达区域后）", rows, n, func(a *acc) [5]float64 { return a.adj })
	fmt.Printf("\n（每题每策略 1 次向量 API 调用，本次共 %d 次）\n", len(cases)*len(rows))
}

// printHeader 打测试集体检表：题量、gold 段数/字数、标签分布。
func printHeader(cases []eval.TestCase) {
	segs, union, multi := 0, 0, 0
	tags := map[string]int{}
	for _, c := range cases {
		segs += len(c.Gold)
		union += c.GoldChars()
		if len(c.Gold) > 1 {
			multi++
		}
		for _, t := range c.Tags {
			tags[t]++
		}
	}
	fmt.Printf("测试集：%s（%d 题 / %d 段 gold / 并集 %d 字 / 平均每题 %d 字 / 多段题 %d 道）\n",
		testsetPath, len(cases), segs, union, union/len(cases), multi)
	fmt.Printf("标签：%s\n", tagLine(tags))
	fmt.Printf("指标：coverage@k = 前 k 条命中的字符区间并集 ∩ gold 区间并集 的字符占比（与切块策略解耦）\n")
	fmt.Printf("      k 阶梯 %v；cov@1 只看头名块，排名敏感（取代单独的 MRR 列）\n", kLadder)
	fmt.Printf("      校正 = 扣掉该策略本身检索不到的区域之后的比例（按标题策略的标题行不在任何块的正文区间里）\n")
	fmt.Printf("候选池：每题 top-20（向量 / BM25 / RRF 三路各自算）\n\n")
}

// tagLine 把标签计数印成 "字面×8  多段×5  …"（按标签名排序，读数稳定）。
func tagLine(tags map[string]int) string {
	names := make([]string, 0, len(tags))
	for t := range tags {
		names = append(names, t)
	}
	sort.Strings(names)
	parts := make([]string, len(names))
	for i, t := range names {
		parts[i] = fmt.Sprintf("%s×%d", t, tags[t])
	}
	return strings.Join(parts, "  ")
}

// printTable 每个策略 × 每条路线一行，k 阶梯作列（值都已按题数平均）。
func printTable(title string, rows []row, n float64, pick func(*acc) [5]float64) {
	fmt.Printf("\n──────── %s ────────\n", title)
	fmt.Printf("%-18s %-5s", "策略", "路线")
	for _, k := range kLadder {
		fmt.Printf("  @%-5d", k)
	}
	fmt.Println()
	for _, r := range rows {
		for _, name := range []string{"向量", "BM25", "RRF"} {
			v := pick(r.cov[name])
			fmt.Printf("%-18s %-5s", fmt.Sprintf("%s(%d块)", r.name, r.chunks), name)
			for _, x := range v {
				fmt.Printf("  %.3f  ", x/n)
			}
			fmt.Println()
		}
	}
}

// ladder 把一题的 k 阶梯覆盖印成一行。
func ladder(vals []float64) string {
	parts := make([]string, len(kLadder))
	for i, k := range kLadder {
		parts[i] = fmt.Sprintf("cov@%d=%.2f", k, vals[i])
	}
	return strings.Join(parts, "  ")
}

// verifyGold 体检：gold 的 [start,end) 必须落在语料里，且切片内容与标注原文逐字一致。
// 标注 spec → JSON → 评测三环之间，这里是唯一的自动保险（防语料/标注漂移）。
func verifyGold(cases []eval.TestCase, docs []*schema.Document) error {
	byID := make(map[string]string, len(docs))
	for _, d := range docs {
		byID[d.ID] = d.Content
	}
	for i, c := range cases {
		for j, g := range c.Gold {
			content, ok := byID[g.Doc]
			if !ok {
				return fmt.Errorf("第 %d 题（%s）第 %d 段 gold 指向语料外的文档 %s", i+1, c.Question, j+1, g.Doc)
			}
			r := []rune(content)
			if g.Start < 0 || g.End > len(r) || g.End <= g.Start {
				return fmt.Errorf("第 %d 题（%s）第 %d 段 gold 区间 [%d,%d) 越界（%s 全长 %d 字）",
					i+1, c.Question, j+1, g.Start, g.End, g.Doc, len(r))
			}
			if got := string(r[g.Start:g.End]); got != g.Text {
				return fmt.Errorf("第 %d 题（%s）第 %d 段 gold 与语料不一致：\n  语料：%s\n  标注：%s",
					i+1, c.Question, j+1, preview(got, 60), preview(g.Text, 60))
			}
		}
	}
	return nil
}

// preview 截断长文本用于报错信息（换行显示为 \n，避免把一行报错撑成十行）。
func preview(s string, maxRunes int) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	r := []rune(s)
	if len(r) <= maxRunes {
		return s
	}
	return string(r[:maxRunes]) + "…"
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

// goldDesc 把一题的 gold 印成 "doc[start,end)、doc[start,end)（n 段 / 并集 m 字）"。
func goldDesc(c eval.TestCase) string {
	var b strings.Builder
	for i, g := range c.Gold {
		if i > 0 {
			b.WriteString("、")
		}
		fmt.Fprintf(&b, "%s[%d,%d)", g.Doc, g.Start, g.End)
	}
	fmt.Fprintf(&b, "（%d 段 / 并集 %d 字）", len(c.Gold), c.GoldChars())
	return b.String()
}
