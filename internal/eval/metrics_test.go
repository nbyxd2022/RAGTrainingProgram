package eval

import (
	"math"
	"testing"
)

func approx(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("got %.4f, want %.4f", got, want)
	}
}

// 同一区域被重叠的块重复召回，覆盖不能超过 1。
func TestCoverageOverlapNotDoubleCounted(t *testing.T) {
	gold := []Span{{Doc: "d", Start: 100, End: 200}}
	retrieved := []Span{
		{Doc: "d", Start: 100, End: 200},
		{Doc: "d", Start: 150, End: 250}, // 只有 200~250 是新的
	}
	approx(t, CoverageAtK(retrieved, gold, 2), 1.0)
}

// k 截断：只看前 k 条。
func TestCoverageKTruncates(t *testing.T) {
	gold := []Span{{Doc: "d", Start: 0, End: 100}}
	retrieved := []Span{
		{Doc: "d", Start: 0, End: 40},
		{Doc: "d", Start: 40, End: 100},
	}
	approx(t, CoverageAtK(retrieved, gold, 1), 0.4)
	approx(t, CoverageAtK(retrieved, gold, 2), 1.0)
}

// 跨文档：不同 doc 互不相干（同下标也不是同一处文字）；漏召回的文档留在分母。
func TestCoverageCrossDoc(t *testing.T) {
	gold := []Span{
		{Doc: "a", Start: 0, End: 100},
		{Doc: "b", Start: 0, End: 100},
	}
	retrieved := []Span{
		{Doc: "a", Start: 0, End: 100},
		{Doc: "c", Start: 0, End: 999}, // 非 gold 文档，不计入分子
	}
	approx(t, CoverageAtK(retrieved, gold, 2), 0.5)
}

// gold 被切成相邻两段、召回块横跨接缝：先把 gold 并集合并，再求交。
func TestCoverageMergeBeforeIntersect(t *testing.T) {
	gold := []Span{{Doc: "d", Start: 0, End: 50}, {Doc: "d", Start: 50, End: 100}}
	retrieved := []Span{{Doc: "d", Start: 25, End: 75}}
	approx(t, CoverageAtK(retrieved, gold, 1), 0.5)
}

// 边界：空召回 / 空 gold / k<=0。
func TestCoverageEdges(t *testing.T) {
	gold := []Span{{Doc: "d", Start: 500, End: 1400}} // Q9 形状：gold 共 900 字
	if got := CoverageAtK(nil, gold, 5); got != 0 {
		t.Errorf("空召回应为 0，got %.4f", got)
	}
	if got := CoverageAtK([]Span{{Doc: "d", Start: 0, End: 10}}, nil, 5); got != 0 {
		t.Errorf("空 gold 应为 0，got %.4f", got)
	}
	if got := CoverageAtK([]Span{{Doc: "d", Start: 0, End: 10}}, gold, 0); got != 0 {
		t.Errorf("k=0 应为 0，got %.4f", got)
	}
	approx(t, CoverageAtK([]Span{{Doc: "d", Start: 500, End: 1000}}, gold, 1), 500.0/900.0)
}
