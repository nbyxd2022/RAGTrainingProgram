package rag

import (
	"context"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// Retrieval 把"原问题 + 检索到的资料"打包在一起。
// 链上的节点只能收到上一个节点的输出——检索完之后问题就"丢"了，
// 而拼提示词需要问题和资料同时在场，所以用这个结构把它们捎着往后传。
type Retrieval struct {
	Question string
	Docs     []*schema.Document
}

// NewQAChain 组装"增强 → 生成"链：
//
//	*Retrieval ──(BuildMessages)──► []*schema.Message ──(ChatModel)──► *schema.Message
//
// 为什么检索不放进链里：
//   - 链是单入单出的流水线，检索节点只能把 []*Document 传给下一个节点，问题会丢；
//     要让"问题"和"资料"两路数据在同一个节点汇合，需要 Graph 的多路输入（阶段 4 再学）
//   - 检索结果在 cmd/ask 已经拿到并展示（调试窗口），放进链里会重复调用，白付一次 embedding 的钱
func NewQAChain(ctx context.Context, cm model.BaseChatModel) (compose.Runnable[*Retrieval, *schema.Message], error) {
	chain := compose.NewChain[*Retrieval, *schema.Message]()

	chain.AppendLambda(compose.InvokableLambda(
		func(ctx context.Context, in *Retrieval) ([]*schema.Message, error) {
			return BuildMessages(in.Question, in.Docs), nil
		},
	))

	chain.AppendChatModel(cm)

	return chain.Compile(ctx)
}
