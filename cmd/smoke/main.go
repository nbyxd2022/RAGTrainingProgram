package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	"github.com/nbyxd2022/RAGTrainingProgram/internal/config"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/llm"
)

func main() {
	ctx := context.Background()

	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("加载配置失败：%v", err)
	}

	embedder, err := llm.NewEmbedder(ctx, cfg)
	if err != nil {
		log.Fatalf("构造 Embedder 失败：%v", err)
	}
	chatModel, err := llm.NewChatModel(ctx, cfg)
	if err != nil {
		log.Fatalf("构造 ChatModel 失败：%v", err)
	}

	failed := false
	if err := smokeEmbedding(ctx, cfg, embedder); err != nil {
		fmt.Printf("  [失败] %v\n\n", err)
		failed = true
	}
	if err := smokeChat(ctx, cfg, chatModel); err != nil {
		fmt.Printf("  [失败] %v\n\n", err)
		failed = true
	}
	if failed {
		os.Exit(1)
	}
	fmt.Println("两项冒烟测试全部通过。")
}

func smokeEmbedding(ctx context.Context, cfg *config.Config, embedder embedding.Embedder) error {
	fmt.Printf("=== 1. embedding 冒烟测试（模型：%s）===\n", cfg.ArkEmbeddingModel)

	text := "检索增强生成（RAG）在生成之前先检索外部知识，缓解大模型的知识过时与幻觉问题。"
	fmt.Printf("输入（%d 字）：%s\n\n", len([]rune(text)), text)

	vecs, err := embedder.EmbedStrings(ctx, []string{text})
	if err != nil {
		return err
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return fmt.Errorf("embedding 返回为空")
	}
	vec := vecs[0]

	fmt.Printf("返回条数   ：%d（与输入条数一致）\n", len(vecs))
	fmt.Printf("向量维度   ：%d\n", len(vec))

	n := 8
	if len(vec) < n {
		n = len(vec)
	}
	fmt.Print("前 8 个分量：")
	for _, v := range vec[:n] {
		fmt.Printf("% .6f  ", v)
	}
	fmt.Println()

	var sumSq float64
	negative := 0
	for _, v := range vec {
		sumSq += v * v
		if v < 0 {
			negative++
		}
	}
	fmt.Printf("L2 模长    ：%.6f\n", math.Sqrt(sumSq))
	fmt.Printf("负分量占比 ：%d/%d\n\n", negative, len(vec))
	return nil
}

func smokeChat(ctx context.Context, cfg *config.Config, chatModel model.ChatModel) error {
	fmt.Printf("=== 2. chat 冒烟测试（模型：%s）===\n", cfg.ArkChatModel)

	resp, err := chatModel.Generate(ctx, []*schema.Message{
		schema.SystemMessage("你是简洁的助手，只回答一句话。"),
		schema.UserMessage("RAG 是什么？"),
	})
	if err != nil {
		return err
	}

	fmt.Println("问题：RAG 是什么？")
	fmt.Printf("回复：%s\n", resp.Content)
	if resp.ResponseMeta != nil && resp.ResponseMeta.Usage != nil {
		u := resp.ResponseMeta.Usage
		fmt.Printf("token 用量：prompt=%d  completion=%d  total=%d\n\n", u.PromptTokens, u.CompletionTokens, u.TotalTokens)
	}
	return nil
}
