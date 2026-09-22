package eval

import (
	"encoding/json"
	"fmt"
	"os"
)

// TestCase 一条评测样本：一个问题 + 标准答案块清单（gold）。
// gold 可能多个：同一个问题有几块都算答对（见幻觉那题）。
type TestCase struct {
	Question string   `json:"question"`
	Gold     []string `json:"gold"`
	Note     string   `json:"note"`
}

// LoadTestset 读取 JSON 测试集并做基本校验。
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
	}
	return cases, nil
}
