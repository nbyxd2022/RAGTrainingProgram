package eval

import (
	"encoding/json"
	"fmt"
	"os"
)

// GoldSpan 一段标准答案：源文档里的字符区间 [Start, End)（rune 下标）+ 原文。
// Text 是标注时引用的原文，加载后由评测入口与语料逐字核对（防标注与语料漂移）。
type GoldSpan struct {
	Doc   string `json:"doc"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Text  string `json:"text"`
}

// TestCase 一条评测样本：一个问题 + 若干段标准答案文本（gold）。
// gold 多段 = 答案分散在多处：带回一段拿一段的分（见"检索方式的演变"那题）。
type TestCase struct {
	Question string     `json:"question"`
	Tags     []string   `json:"tags"`
	Gold     []GoldSpan `json:"gold"`
	Note     string     `json:"note"`
}

// Spans 把一题的 gold 转成覆盖率计算用的区间列表。
func (c TestCase) Spans() []Span {
	out := make([]Span, 0, len(c.Gold))
	for _, g := range c.Gold {
		out = append(out, Span{Doc: g.Doc, Start: g.Start, End: g.End})
	}
	return out
}

// GoldChars 一题 gold 的总字符数（多段按并集算）。
func (c TestCase) GoldChars() int {
	return UnionLen(c.Spans())
}

// LoadTestset 读取 JSON 测试集并做基本校验。
// 结构见 eval/testset.json（由 eval/testset.spec.txt 经 cmd/sections -emit 编译而来）。
func LoadTestset(path string) ([]TestCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cases []TestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("解析 %s 失败：%w", path, err)
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("%s 里没有评测样本", path)
	}
	for i, c := range cases {
		if c.Question == "" || len(c.Gold) == 0 {
			return nil, fmt.Errorf("第 %d 条缺少 question 或 gold", i+1)
		}
		for j, g := range c.Gold {
			if g.Doc == "" || g.Start < 0 || g.End <= g.Start || g.Text == "" {
				return nil, fmt.Errorf("第 %d 条（%s）第 %d 段 gold 不完整（doc/start/end/text 都要有）", i+1, c.Question, j+1)
			}
		}
	}
	return cases, nil
}
