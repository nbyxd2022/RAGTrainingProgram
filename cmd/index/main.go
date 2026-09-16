package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/nbyxd2022/RAGTrainingProgram/internal/config"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/llm"
	"github.com/nbyxd2022/RAGTrainingProgram/internal/rag"
)

const indexPath = ".data/vectors.gob"

func main() {
	size := flag.Int("size", 500, "每块目标长度（rune）")
	overlap := flag.Int("overlap", 50, "相邻块重叠长度（rune）")
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

	docs, err := rag.LoadCorpus(ctx, "corpus")
	if err != nil {
		log.Fatal(err)
	}
	chunks, err := (&rag.Splitter{ChunkSize: *size, Overlap: *overlap}).Transform(ctx, docs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("语料：%d 篇 → %d 块，开始向量化（模型：%s）\n", len(docs), len(chunks), cfg.ArkEmbeddingModel)
	fmt.Println("多模态接口逐条调用（适配器内 5 并发），预计 1~3 分钟；结果会存到 " + indexPath)

	start := time.Now()
	idx, err := rag.BuildIndex(ctx, chunks, embedder, func(done, total int) {
		if done%100 == 0 || done == total {
			fmt.Printf("  进度 %d/%d（%.0fs）\n", done, total, time.Since(start).Seconds())
		}
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := idx.Save(indexPath); err != nil {
		log.Fatal(err)
	}

	fi, err := os.Stat(indexPath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("完成：%d 条，用时 %.0fs，索引 %s（%.1f MB）\n", len(idx.Entries), time.Since(start).Seconds(), indexPath, float64(fi.Size())/1024/1024)
}
