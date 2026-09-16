package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"github.com/nbyxd2022/RAGTrainingProgram/internal/config"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/llm"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

func main() {
	q := flag.String("q", "", "问题")
	k := flag.Int("k", 3, "返回条数 top-k")
	flag.Parse()
	if *q == "" {
		log.Fatal("用法：go run ./cmd/ask -q \"你的问题\" [-k 3]")
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

	idx, err := rag.LoadIndex(".data/vectors.gob")
	if err != nil {
		log.Fatalf("加载索引失败（先运行 go run ./cmd/index）：%v", err)
	}

	r := &rag.MemoryRetriever{Index: idx, Embedder: embedder, TopK: *k}
	docs, err := r.Retrieve(ctx, *q)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("问题：%s\n索引：%d 块，返回 top-%d\n\n", *q, len(idx.Entries), *k)
	for i, d := range docs {
		score, _ := d.MetaData["score"].(float64)
		content := []rune(d.Content)
		if len(content) > 200 {
			content = append(content[:200], '…')
		}
		fmt.Printf("[%d] score=%.4f  %s（%v）\n%s\n\n", i+1, score, d.ID, d.MetaData["source"], string(content))
	}

	chatModel, err := llm.NewChatModel(ctx, cfg)
	if err != nil {
		log.Fatal(err)
	}
	qa, err := rag.NewQAChain(ctx, chatModel)
	if err != nil {
		log.Fatal(err)
	}
	reply, err := qa.Invoke(ctx, &rag.Retrieval{Question: *q, Docs: docs})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("──────────────── 回答 ────────────────")
	fmt.Println(reply.Content)
	fmt.Printf("=========================================\n")
	if reply.ReasoningContent != "" {
		fmt.Printf("思考内容：%s\n", reply.ReasoningContent)
		fmt.Printf("\n（附带思考内容 %d 字）\n", len([]rune(reply.ReasoningContent)))
	}
	fmt.Printf("=========================================\n")
	if m := reply.ResponseMeta; m != nil && m.Usage != nil {
		u := m.Usage
		fmt.Printf("\ntoken 用量：prompt=%d completion=%d total=%d", u.PromptTokens, u.CompletionTokens, u.TotalTokens)
		if rt := u.CompletionTokensDetails.ReasoningTokens; rt > 0 {
			fmt.Printf("（其中推理 token %d）", rt)
		}
		fmt.Println()
	}
}
