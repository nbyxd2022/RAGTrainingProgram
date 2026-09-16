package main

import (
	"context"
	"fmt"
	"log"

	"github.com/nbyxd2022/RAGTrainingProgram/internal/config"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/llm"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

func main() {
	ctx := context.Background()

	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}
	embedder, err := llm.NewEmbedder(ctx, cfg)
	if err != nil {
		log.Fatal(err)
	}

	texts := []string{
		"检索增强生成（RAG）是一种让大模型先查资料再回答的技术。",
		"RAG 通过检索外部知识库来减少大模型的幻觉。",
		"红烧肉的做法：五花肉切块，冷水下锅焯水，加冰糖炒出糖色。",
	}
	vecs, err := embedder.EmbedStrings(ctx, texts)
	if err != nil {
		log.Fatal(err)
	}
	if len(vecs) != len(texts) {
		log.Fatalf("返回 %d 条向量，期望 %d 条", len(vecs), len(texts))
	}

	fmt.Println("三个句子：")
	for i, t := range texts {
		fmt.Printf("  %d. %s\n", i+1, t)
	}
	base := rag.ToFloat32(vecs[0])
	fmt.Println()
	fmt.Printf("cos(1, 2) 语义相关   = %.4f\n", rag.Cosine(base, rag.ToFloat32(vecs[1])))
	fmt.Printf("cos(1, 3) 语义无关   = %.4f\n", rag.Cosine(base, rag.ToFloat32(vecs[2])))
	fmt.Println("\n预期：相关那对明显更高。如果两者接近甚至反了，回去检查 Cosine 的实现。")
}
