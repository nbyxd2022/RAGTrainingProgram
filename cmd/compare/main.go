package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/cloudwego/eino/schema"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/config"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/llm"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

func main() {
	q := flag.String("q", "", "问题")
	k := flag.Int("k", 5, "最终展示条数")
	n := flag.Int("n", 20, "每路召回的候选池大小（RRF 的输入）")
	idxPath := flag.String("idx", ".data/vectors.gob", "索引文件路径（如 .data/idx-heading.gob）")
	flag.Parse()
	if *q == "" {
		log.Fatal("用法：go run ./cmd/compare -q \"你的问题\" [-k 5] [-n 20]")
	}
	if *n < *k {
		*n = *k
	}

	ctx := context.Background()
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}
	embedder, err := llm.NewEmbedder(ctx, cfg)
	if err != nil {
		log.Fatal(err)
	}
	idx, err := rag.LoadIndex(*idxPath)
	if err != nil {
		log.Fatalf("加载索引失败（先运行 go run ./cmd/index）：%v", err)
	}

	chunks := idx.ToDocuments()
	fmt.Printf("问题：%s\n索引：%s（%d 块，候选池 top-%d，展示 top-%d）\n\n", *q, *idxPath, len(chunks), *n, *k)

	// 路线 1：纯向量。唯一要调 API、花钱、耗时在秒级的一步。
	vr := &rag.MemoryRetriever{Index: idx, Embedder: embedder, TopK: *n}
	t0 := time.Now()
	vecDocs, err := vr.Retrieve(ctx, *q)
	if err != nil {
		log.Fatal(err)
	}
	vecTime := time.Since(t0)

	// 路线 2：BM25。纯 CPU，不花钱，建索引 + 检索都在毫秒级。
	t1 := time.Now()
	bm := rag.BuildBM25(chunks)
	bmBuildTime := time.Since(t1)
	t2 := time.Now()
	bmDocs := bm.Search(*q, *n)
	bmTime := time.Since(t2)

	// 路线 3：RRF 融合。k=60 是原论文以来的惯例值。
	t3 := time.Now()
	fused := rag.FuseDocuments([][]*schema.Document{vecDocs, bmDocs}, 60)
	fuseTime := time.Since(t3)

	printList("路线 1 · 纯向量", vecTime, vecDocs, *k)
	printList("路线 2 · BM25", bmBuildTime+bmTime, bmDocs, *k)
	printList("路线 3 · RRF 混合", fuseTime, fused, *k)

	fmt.Printf("耗时：向量检索 %s（含一次 API 调用）｜ BM25 建索引 %s + 检索 %s ｜ RRF %s\n",
		fmtDur(vecTime), fmtDur(bmBuildTime), fmtDur(bmTime), fmtDur(fuseTime))
}

func printList(title string, dur time.Duration, docs []*schema.Document, k int) {
	fmt.Printf("──────── %s ｜ 耗时 %s ────────\n", title, fmtDur(dur))
	if len(docs) == 0 {
		fmt.Println("（无命中）")
		fmt.Println()
		return
	}
	if len(docs) > k {
		docs = docs[:k]
	}
	for i, d := range docs {
		score, _ := d.MetaData["score"].(float64)
		fmt.Printf("[%d] score=%.4f  %s（%v）\n    %s\n", i+1, score, d.ID, d.MetaData["source"], preview(d.Content, 80))
	}
	fmt.Println()
}

// preview 把多行内容压成一行并截断，方便三路结果并排对比。
func preview(s string, maxRunes int) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > maxRunes {
		return string(r[:maxRunes]) + "…"
	}
	return s
}

func fmtDur(d time.Duration) string {
	if d < time.Millisecond {
		return d.Round(time.Microsecond).String()
	}
	return d.Round(time.Millisecond).String()
}
