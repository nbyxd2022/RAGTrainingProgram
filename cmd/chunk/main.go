package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"
	"unicode/utf8"

	"github.com/cloudwego/eino/schema"

	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

func main() {
	strategy := flag.String("strategy", "fixed", "切块策略：fixed ｜ heading ｜ heading-nobc（标题链不进正文）")
	size := flag.Int("size", 500, "fixed 策略：每块目标长度（rune）")
	overlap := flag.Int("overlap", 50, "fixed 策略：相邻块重叠长度（rune）")
	maxLen := flag.Int("maxlen", 800, "heading 策略：单块正文上限（rune）")
	grep := flag.String("grep", "", "只罗列内容含该子串的块（找评测 gold 块用）")
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

	transformer, err := rag.NewTransformer(*strategy, *size, *overlap, *maxLen)
	if err != nil {
		log.Fatal(err)
	}
	chunks, err := transformer.Transform(ctx, docs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("切块：策略 %s → 共 %d 块\n", *strategy, len(chunks))
	if len(chunks) == 0 {
		return
	}

	total, shortest, longest := 0, 1<<30, 0
	for _, c := range chunks {
		n := utf8.RuneCountInString(c.Content)
		total += n
		if n < shortest {
			shortest = n
		}
		if n > longest {
			longest = n
		}
	}
	fmt.Printf("块长度：最短 %d / 最长 %d / 平均 %.0f 字\n", shortest, longest, float64(total)/float64(len(chunks)))

	if *grep != "" {
		lower := strings.ToLower(*grep)
		found := 0
		for _, c := range chunks {
			if strings.Contains(strings.ToLower(c.Content), lower) {
				found++
				fmt.Printf("  %s （%d 字）  %s\n", c.ID, utf8.RuneCountInString(c.Content), preview(c.Content, 60))
			}
		}
		fmt.Printf("\n含 %q 的块：%d 个（块 ID 与同参数构建的索引一致）\n", *grep, found)
		return
	}

	fmt.Printf("\n=== 第一块 %s（%d 字）===\n%s\n", chunks[0].ID, utf8.RuneCountInString(chunks[0].Content), chunks[0].Content)

	if len(chunks) > 1 {
		fmt.Println("\n=== 相邻两块边界（第 1 块结尾 / 第 2 块开头）===")
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

// preview 压掉换行并截断，方便一眼扫过每块的开头。
func preview(s string, maxRunes int) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > maxRunes {
		return string(r[:maxRunes]) + "…"
	}
	return s
}
