package rag

import (
	"math"
	"sort"

	"github.com/cloudwego/eino/schema"
)

// BM25 的两个标准参数：k1 控制词频饱和速度，b 控制长度归一化强度。
// k1越大，出现次数越多的词条对分数的贡献越大；k1越小，出现次数越多的词条对分数的贡献越小。
// 长文档天然占便宜（词多，撞上查询词的机会大）——所以按文档长度惩罚。
// 参数 b 控制惩罚强度（b=0 不惩罚，b=1 完全按相对长度惩罚）。
const (
	BM25K1 = 1.5
	BM25B  = 0.75
)

type bm25Doc struct {
	freq   map[string]int // 词条 → 在该块里出现的次数
	length int            // 该块的词条总数（用于长度归一化）
}

// BM25Index 手写 BM25 稀疏检索索引：纯 CPU 构建，不调 API、不花钱，也不需要落盘。
type BM25Index struct {
	docs   []*schema.Document // 与 terms 一一对应的原文块
	terms  []bm25Doc
	df     map[string]int // 词条 → 出现在多少个块里（document frequency）
	avgLen float64        // 所有块的平均词条数
}

// BuildBM25 对全部块建索引：分词、统计词频与 df。
func BuildBM25(chunks []*schema.Document) *BM25Index {
	ix := &BM25Index{
		docs:  chunks,
		terms: make([]bm25Doc, len(chunks)),
		df:    map[string]int{},
	}
	total := 0
	for i, c := range chunks {
		toks := Tokenize(c.Content)
		freq := make(map[string]int, len(toks))
		for _, t := range toks {
			freq[t]++
		}
		ix.terms[i] = bm25Doc{freq: freq, length: len(toks)}
		total += len(toks)
		for t := range freq {
			ix.df[t]++
		}
	}
	if n := len(chunks); n > 0 {
		ix.avgLen = float64(total) / float64(n)
	}
	return ix
}

// Search 返回按 BM25 分数降序的 top-k 块。
// 分数为 0 的块直接丢掉：一个查询词都对不上的块，没有排在结果里的意义。
func (ix *BM25Index) Search(query string, topK int) []*schema.Document {
	qterms := Tokenize(query)

	type hit struct {
		idx   int
		score float64
	}
	hits := make([]hit, 0, len(ix.docs))
	for i := range ix.docs {
		s := ix.score(qterms, ix.terms[i])
		if s > 0 {
			hits = append(hits, hit{idx: i, score: s})
		}
	}
	sort.Slice(hits, func(a, b int) bool { return hits[a].score > hits[b].score })

	if topK > len(hits) {
		topK = len(hits)
	}
	out := make([]*schema.Document, 0, topK)
	for _, h := range hits[:topK] {
		d := ix.docs[h.idx]
		meta := make(map[string]any, len(d.MetaData)+1)
		for k, v := range d.MetaData {
			meta[k] = v
		}
		meta["score"] = h.score
		out = append(out, &schema.Document{ID: d.ID, Content: d.Content, MetaData: meta})
	}
	return out
}

// score 计算查询词条集合与一篇文档的 BM25 分数 —— 步骤 2A 任务 2，由你实现。
//
// 公式（对 query 里每个词条 t 求和）：
//
//	score = Σ_t  IDF(t) · tf·(k1+1) / (tf + k1·(1 - b + b·len/avgdl))
//	IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
//
// 要求：
//  1. tf 取 d.freq[t]；tf == 0 的项直接跳过（分子也是 0，跳过更省）
//  2. df 取 ix.df[t]；N = len(ix.docs)；avgdl = ix.avgLen；k1/b 用 BM25K1 / BM25B
//  3. 全程 float64（提示：会用到 math.Log）

// tf = term frequency，词频：一个词在某一篇文档（块）里出现了几次。它freq map 存的东西
// df = document frequency，文档频率：一个词在多少篇文档（块）里出现过。它 ix.df map 存的东西
// k1 控制词频饱和速度，b 控制长度归一化强度。
// k1越大，出现次数越多的词条对分数的贡献越大；k1越小，出现次数越多的词条对分数的贡献越小。
// 长文档天然占便宜（词多，撞上查询词的机会大）——所以按文档长度惩罚。
// 参数 b 控制惩罚强度（b=0 不惩罚，b=1 完全按相对长度惩罚）。len = 文档词条数，avgdl = 平均文档长度（词条数）
// N在 IDF 公式里扮演的角色：算这个词“稀不稀罕”；核心就是那个比值：“没它的块”比“有它的块”。没它的块越多 → 词越稀罕 → IDF 越大
func (ix *BM25Index) score(queryTerms []string, d bm25Doc) float64 {

	N := float64(len(ix.docs))
	var totalScore float64

	for _, t := range queryTerms {
		tf := float64(d.freq[t])
		if tf == 0 {
			continue
		}
		idf := math.Log(1 + (N-float64(ix.df[t])+0.5)/(float64(ix.df[t])+0.5))
		length := float64(d.length)
		score := idf * tf * (BM25K1 + 1) / (tf + BM25K1*(1-BM25B+BM25B*length/ix.avgLen))
		totalScore += score
	}
	return totalScore
}
