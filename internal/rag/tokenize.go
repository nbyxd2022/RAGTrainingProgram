package rag

import (
	"strings"
	"unicode"
)

// Tokenize 把文本切成 BM25 用的"词条" —— 步骤 2A 任务 1，由你实现。
//
// 中文没有空格，按空白切行不通。我们采用"字符二元组（bigram）"方案：
// 相邻两个字符组成一个词条。"检索增强" → "检索"、"索增"、"增强"。
// 查询和文档用同一套切法，字面命中依然可靠。
//
// 要求：
//  1. 先把文本转小写（让 "BM25" 和 "bm25" 对得上）
//  2. 转成 rune 数组遍历（不要按 byte），把相邻两个 rune 拼成一个词条
//  3. 两个 rune 里只要有一个是空白字符（空格/换行等，unicode.IsSpace），就跳过这个词条
//  4. 文本不足 2 个 rune 时返回空切片（nil 也可以）
//
// 例子："RAG 是什么" → ["ra", "ag", "是什", "什么"]
// （"g " 和 " 是" 因为跨空白被丢掉，"ra"/"ag" 来自小写化后的 "rag"）
func Tokenize(text string) []string {

	lowerText := strings.ToLower(text)

	runes := []rune(lowerText)

	var tokens []string

	if len(runes) < 2 {
		return nil
	}
	for i := 0; i < len(runes)-1; i++ {
		if !unicode.IsSpace(runes[i]) && !unicode.IsSpace(runes[i+1]) {
			// tokens = append(tokens, string(runes[i])+string(runes[i+1]))
			tokens = append(tokens, string(runes[i:i+2]))
		}
	}

	return tokens
}
