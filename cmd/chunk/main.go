package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"unicode/utf8"

	"github.com/cloudwego/eino/schema"

	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

func main() {
	size := flag.Int("size", 500, "每块目标长度（rune）")
	overlap := flag.Int("overlap", 50, "相邻块重叠长度（rune）")
	flag.Parse()

	ctx := context.Background()

	docs, err := rag.LoadCorpus(ctx, "corpus")
	if err != nil {
		log.Fatal(err)
	}
	totalChars := 0
	for _, d := range docs {
		totalChars += utf8.RuneCountInString(d.Content)
	}
	fmt.Printf("加载：%d 篇文档，共 %d 字\n", len(docs), totalChars)

	chunks, err := (&rag.Splitter{ChunkSize: *size, Overlap: *overlap}).Transform(ctx, docs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("切块：size=%d overlap=%d → 共 %d 块\n", *size, *overlap, len(chunks))
	if len(chunks) == 0 {
		return
	}

	total, minLen, maxLen := 0, 1<<30, 0
	for _, c := range chunks {
		n := utf8.RuneCountInString(c.Content)
		total += n
		if n < minLen {
			minLen = n
		}
		if n > maxLen {
			maxLen = n
		}
	}
	fmt.Printf("块长度：最短 %d / 最长 %d / 平均 %.0f 字\n", minLen, maxLen, float64(total)/float64(len(chunks)))

	fmt.Printf("\n=== 第一块 %s（%d 字）===\n%s\n", chunks[0].ID, utf8.RuneCountInString(chunks[0].Content), chunks[0].Content)

	if len(chunks) > 1 {
		fmt.Println("\n=== 重叠演示：相邻两块共享的文本 ===")
		printBoundary(chunks[0], chunks[1], 60)
	}
}

func printBoundary(a, b *schema.Document, n int) {
	ra, rb := []rune(a.Content), []rune(b.Content)
	if len(ra) > n {
		ra = ra[len(ra)-n:]
	}
	if len(rb) > n {
		rb = rb[:n]
	}
	fmt.Printf("%s 结尾：…%s\n", a.ID, string(ra))
	fmt.Printf("%s 开头：%s…\n", b.ID, string(rb))
}
