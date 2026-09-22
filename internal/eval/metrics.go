package eval

import (
	"slices"
	"sort"
)

// RecallAtK 计算一个查询的 recall@k —— 评测任务 1，由你实现。
//
// 入参：
//
//	ranked：某条检索路线返回的块 ID，按相关性从高到低
//	gold：  该题的标准答案块清单（可能多个）
//	k：     只看前 k 名
//
// 定义：前 k 名里出现过的 gold 块数量 ÷ len(gold)。
//
// 要求：
//  1. gold 先用 map[string]struct{} 变成集合再查（比双层循环快，也好读）
//  2. k 可能大于 len(ranked)：先截断再数，别越界
//  3. k <= 0 返回 0；返回 0~1 之间的 float64
//  4. 统计"出现过"即可，同一块出现多次只算一次（集合天然保证）
func RecallAtK(ranked, gold []string, k int) float64 {
	if k <= 0 || len(gold) == 0 {
		return 0
	}
	if k > len(ranked) {
		k = len(ranked)
	}
	recallNum := 0
	rankedSlice := ranked[:k]

	//先把 gold 转成集合，方便查找
	goldSet := make(map[string]struct{}, len(gold))
	for _, id := range gold {
		goldSet[id] = struct{}{}
	}

	for id := range goldSet {
		if slices.Contains(rankedSlice, id) {
			recallNum++
		}
	}
	return float64(recallNum) / float64(len(gold))

}

// ReciprocalRank 计算一个查询的倒排排名（RR）—— 评测任务 2，由你实现。
//
// 定义：第一个命中的 gold 块排在第 r 名（r 从 1 计），RR = 1/r；
// 整个 ranked 里一个都没命中 → 0。
//
// 要求：
//
//  1. 只能在 gold 集合里找
//  2. 注意切片下标从 0 开始，r = 下标 + 1
//  3. 返回 float64；gold 为空时返回 0（加载器会拦住这种情况，不用特意处理）
func ReciprocalRank(ranked, gold []string) float64 {

	if len(gold) == 0 {
		return 0
	}
	for i, id := range ranked {
		if slices.Contains(gold, id) {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

// Span 源文档里的一段字符区间 [Start, End)，rune 下标。
// 切块时由 splitter 写进块的 meta（start/end），检索结果全链路保留。
type Span struct {
	Doc        string
	Start, End int
}

// CoverageAtK 计算"前 k 条检索结果覆盖了 gold 答案区域的多大比例" —— 2B 核心任务 2，由你实现。
//
// 背景：换切块策略后块 ID 全变，"按 ID 数命中"的 recall@k 没法跨策略比较；
// 但"答案在原文里占哪段字符"与切块方式无关 —— 这是与切块解耦的公平尺子。
//
// 入参：
//
//	retrieved：某条检索路线返回的字符区间，按相关性降序（同一 doc 可能多条且互相重叠）
//	gold：     该题 gold 的字符区间（同一 doc 可能多段，多段之间可能相邻/重叠）
//	k：        只看前 k 条
//
// 定义：coverage = |(前 k 条的并集) ∩ (gold 的并集)| / |gold 的并集|（按 rune 字符数）
//
// 要求：
//  1. k <= 0 或 gold 为空 → 返回 0；k > len(retrieved) 时按 len 截断
//  2. 先把 retrieved 的区间并集算出来，再和 gold 并集求交 —— 直接逐条求交再加会把
//     重叠部分重复计数（overlap=50 的相邻块、同块被多路命中等都会重叠）
//  3. 按 doc 分组计算，不同 doc 的区间互不相干
//  4. gold 也要先取并集（题目的多个 gold 块区间相邻/重叠很常见）
//
// 提示：同一 doc 内把区间按 Start 排序后单次扫描即可合并区间 —— 当前区间的 End 若 >=
// 下一区间的 Start 就吞并（End 取两者较大值）；区间求交同理。
func CoverageAtK(retrieved, gold []Span, k int) float64 {
	if k <= 0 || len(gold) == 0 {
		return 0
	}
	if k > len(retrieved) {
		k = len(retrieved)
	}

	goldByDoc := groupByDoc(gold)
	retByDoc := groupByDoc(retrieved[:k])

	totalGold, covered := 0, 0
	for doc, gs := range goldByDoc {
		merged := MergeSpans(gs)
		totalGold += spanLen(merged)
		rs, ok := retByDoc[doc]
		if !ok {
			continue // 这篇文档一条都没召回：分子 +0，但它的长度留在分母里
		}
		covered += intersectSpanSets(MergeSpans(rs), merged)
	}
	if totalGold == 0 {
		return 0
	}
	return float64(covered) / float64(totalGold)
}

// groupByDoc 按文档拆抽屉：只有同一篇文档里的区间才允许合并/求交。
func groupByDoc(spans []Span) map[string][]Span {
	byDoc := make(map[string][]Span)
	for _, s := range spans {
		if s.End <= s.Start {
			continue
		}
		byDoc[s.Doc] = append(byDoc[s.Doc], s)
	}
	return byDoc
}

// MergeSpans 返回区间并集（不修改入参）：按 Start 排序后单次扫描，能接上就吞并。
func MergeSpans(spans []Span) []Span {
	sorted := slices.Clone(spans)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Start != sorted[j].Start {
			return sorted[i].Start < sorted[j].Start
		}
		return sorted[i].End < sorted[j].End
	})
	merged := sorted[:0:0]
	for _, s := range sorted {
		if n := len(merged); n > 0 && s.Start <= merged[n-1].End {
			merged[n-1].End = max(merged[n-1].End, s.End)
			continue
		}
		merged = append(merged, s)
	}
	return merged
}

// UnionLen 区间并集的总字符数（同一文档内先合并再求和；跨文档本就不可能重叠）。
func UnionLen(spans []Span) int {
	n := 0
	for _, gs := range groupByDoc(spans) {
		n += spanLen(MergeSpans(gs))
	}
	return n
}

// IntersectLen 两个区间集合的交集总长（按文档分组，文档内先各自合并再求交）。
func IntersectLen(a, b []Span) int {
	aByDoc, bByDoc := groupByDoc(a), groupByDoc(b)
	n := 0
	for doc, as := range aByDoc {
		bs, ok := bByDoc[doc]
		if !ok {
			continue
		}
		n += intersectSpanSets(MergeSpans(as), MergeSpans(bs))
	}
	return n
}

// spanLen 区间并集的总字符数。
func spanLen(spans []Span) int {
	n := 0
	for _, s := range spans {
		n += s.End - s.Start
	}
	return n
}

// intersectSpanSets 两个已合并区间的交集总长（双指针线性扫描）。
func intersectSpanSets(a, b []Span) int {
	n := 0
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		lo := max(a[i].Start, b[j].Start)
		hi := min(a[i].End, b[j].End)
		if lo < hi {
			n += hi - lo
		}
		if a[i].End < b[j].End {
			i++
		} else {
			j++
		}
	}
	return n
}
