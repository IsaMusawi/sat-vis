package main

import (
	"log"

	"github.com/gin-gonic/gin"

	"sat-vis/internal/config"
	"sat-vis/internal/controller"
	"sat-vis/internal/grpc_client"
	"sat-vis/internal/service"
	"sat-vis/internal/storage"
)

func main() {
	// 1. Load Config
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// 2. Initialize gRPC Client (External Adapter)
	grpcClient, err := grpc_client.NewTileClient(cfg.Grpc.PythonWorkerAddress)
	if err != nil {
		log.Fatalf("Failed to connect to Python: %v", err)
	}
	defer grpcClient.Close()

	// 3. Initialize Services (Business Logic)
	// Dependency Injection: Service gets Config
	tilerService := service.NewTilerService(cfg)
	storageClient, err := storage.NewStorageClient(cfg)
	if err != nil {
		log.Fatalf("Failed to connect to storage: %v", err)
	}

	// 4. Initialize Controller (HTTP Layer)
	// Dependency Injection: Controller gets Service and gRPC Client
	satController := controller.NewSatelliteController(tilerService, grpcClient, storageClient)

	// 5. Setup Router (Gin)
	r := gin.Default()

	// API Routes
	r.POST("/upload", satController.UploadAndAnalyze)

	// 6. Run Server
	log.Printf("🚀 Server starting on port %s", cfg.Server.Port)
	if err := r.Run(":" + cfg.Server.Port); err != nil {
		log.Fatalf("Failed to run server: %v", err)
	}
}