package llm

import (
	"context"

	arkembedding "github.com/cloudwego/eino-ext/components/embedding/ark"
	arkmodel "github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"

	"github.com/nbyxd2022/RAGTrainingProgram/internal/config"
)

func NewChatModel(ctx context.Context, cfg *config.Config) (model.ChatModel, error) {
	return arkmodel.NewChatModel(ctx, &arkmodel.ChatModelConfig{
		APIKey: cfg.ArkAPIKey,
		Model:  cfg.ArkChatModel,
	})
}

func NewEmbedder(ctx context.Context, cfg *config.Config) (embedding.Embedder, error) {
	// 方舟在售的向量化模型是多模态系列（doubao-embedding-vision-*），必须走 multi_modal_api 端点
	apiType := arkembedding.APITypeMultiModal
	return arkembedding.NewEmbedder(ctx, &arkembedding.EmbeddingConfig{
		APIKey:  cfg.ArkAPIKey,
		Model:   cfg.ArkEmbeddingModel,
		APIType: &apiType,
	})
}
