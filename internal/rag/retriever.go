package rag

import (
	"context"
	"math"
	"sort"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
)

var _ retriever.Retriever = (*MemoryRetriever)(nil)

// MemoryRetriever 在内存里做暴力向量检索：把 query 编码成向量，
// 和索引里每一块的向量算 cosine，取出分数最高的若干条。
type MemoryRetriever struct {
	Index    *VectorIndex
	Embedder embedding.Embedder
	TopK     int // 默认返回条数；调用方可以用 retriever.WithTopK 覆盖
}

// Retrieve 实现 Eino 的 retriever.Retriever 接口 —— 步骤 4 由你实现。
//
// 要求：
//  1. 先用 retriever.GetCommonOptions(&retriever.Options{TopK: &r.TopK}, opts...) 合并调用方选项
//  2. 把 query 编码成向量：优先用 options.Embedding（没传就用 r.Embedder），复用索引侧的 ToFloat32
//  3. 和 Index.Entries 里每条算 Cosine，按分数从高到低排序
//  4. 取前 options.TopK 条；若设置了 options.ScoreThreshold，把低于它的丢掉
//  5. 返回 []*schema.Document，MetaData 里保留 source / chunk_index，再加一个 "score"（float64）
func (r *MemoryRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {

	retrieveOpts := retriever.GetCommonOptions(&retriever.Options{TopK: &r.TopK}, opts...)

	var QueryVectorFloat32 []float32

	// if retrieveOpts.Embedding != nil {
	// 	QueryVector, err := retrieveOpts.Embedding.EmbedStrings(ctx, []string{query})
	// 	if err != nil {
	// 		return nil, err
	// 	}
	// 	QueryVectorFloat32 = ToFloat32(QueryVector[0])
	// } else {
	// 	QueryVector, err := r.Embedder.EmbedStrings(ctx, []string{query})
	// 	if err != nil {
	// 		return nil, err
	// 	}
	// 	QueryVectorFloat32 = ToFloat32(QueryVector[0])

	// }

	emb := r.Embedder
	if retrieveOpts.Embedding != nil {
		emb = retrieveOpts.Embedding
	}
	QueryVector, err := emb.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	QueryVectorFloat32 = ToFloat32(QueryVector[0])

	var results []*schema.Document

	for _, entry := range r.Index.Entries {
		score := Cosine(QueryVectorFloat32, entry.Vector)
		//不能直接修改 entry.MetaData["score"] = score，这样会污染原数据entry
		newMetaData := make(map[string]any, len(entry.MetaData)+1)
		for k, v := range entry.MetaData {
			newMetaData[k] = v
		}
		newMetaData["score"] = score

		results = append(results, &schema.Document{
			ID:       entry.ID,
			Content:  entry.Content,
			MetaData: newMetaData,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].MetaData["score"].(float64) > results[j].MetaData["score"].(float64)
	})

	if retrieveOpts.ScoreThreshold != nil {
		threshold := *retrieveOpts.ScoreThreshold
		filteredResults := make([]*schema.Document, 0, len(results))
		for _, doc := range results {
			if doc.MetaData["score"].(float64) >= threshold {
				filteredResults = append(filteredResults, doc)
			}
		}
		results = filteredResults
	}

	return results[:min(*retrieveOpts.TopK, len(results))], nil

}

// Cosine 计算两个向量的余弦相似度 —— 步骤 4 由你实现。
//
// 要求：
//  1. cos = dot(a, b) / (|a| * |b|)
//  2. 任一向量为空、或模长为 0 时返回 0
//  3. 用 float64 累加：float32 累 2048 项的误差没必要承担
func Cosine(a, b []float32) float64 {

	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	var dotProduct float64
	var normA float64
	var normB float64

	for i := 0; i < len(a) && i < len(b); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))

}
