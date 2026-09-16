package rag

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/schema"
)

var _ document.Transformer = (*Splitter)(nil)

// Splitter 定长 + 重叠切块器，实现 Eino 的 document.Transformer 接口。
type Splitter struct {
	ChunkSize int // 每块目标长度（rune 数）
	Overlap   int // 相邻块重叠长度（rune 数）
}

func (s *Splitter) Transform(ctx context.Context, src []*schema.Document, opts ...document.TransformerOption) ([]*schema.Document, error) {
	var out []*schema.Document
	for _, doc := range src {
		for i, piece := range SplitText(doc.Content, s.ChunkSize, s.Overlap) {
			meta := make(map[string]any, len(doc.MetaData)+1)
			for k, v := range doc.MetaData {
				meta[k] = v
			}
			meta["chunk_index"] = i
			out = append(out, &schema.Document{
				ID:       fmt.Sprintf("%s#%d", doc.ID, i),
				Content:  piece,
				MetaData: meta,
			})
		}
	}
	return out, nil
}

// SplitText 把一段文本切成块 —— 步骤 3 由你实现。
//
// 要求：
//  1. 按 rune（字符）切，不要按 byte（中文一个字占 3 个字节，按 byte 切会出现乱码）
//  2. 步长 = chunkSize - overlap：相邻两块共享 overlap 个字符
//  3. 最后一块不足 chunkSize 也要保留（除非为空）
//  4. chunkSize <= 0、overlap < 0、overlap >= chunkSize 时返回空切片
func SplitText(text string, chunkSize, overlap int) []string {
	if chunkSize <= 0 || overlap < 0 || overlap >= chunkSize {
		// fmt.Println("Invalid chunkSize or overlap")
		return nil
	}
	step := chunkSize - overlap
	var chunks []string
	runes := []rune(text)
	var subrunes []rune
	for i := 0; i < len(runes); i += step {
		subrunes = subrunes[:0] // 清空子切片
		for j := i; j < i+chunkSize && j < len(runes); j++ {
			subrunes = append(subrunes, runes[j])
		}
		chunks = append(chunks, string(subrunes))

		//程序可以两写完
		//end := min(i+chunkSize, len(runes))
		//chunks = append(chunks, string(runes[i:end]))

	}
	//panic("SplitText 尚未实现")
	return chunks
}
