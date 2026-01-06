package grpc_client

import (
	"context"
	"time"

	// IMPORT TRICK: This imports the code YOU generated earlier.
	// We alias it as 'pb' (ProtoBuf) to make it easier to type.
	pb "sat-vis/pkg/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// NewClient creates the connection.
// address: "localhost:50051"
func NewTileClient(address string) (*TileClient, error) {
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	client := pb.NewImageAnalyzerClient(conn)

	return &TileClient{service: client, conn: conn}, nil
}

func (c *TileClient) Close() error {
	return c.conn.Close()
}

func (c *TileClient) AnalyzeTile(ctx context.Context, id string, imageData []byte) (*pb.TileResponse, error) {
	// 1. Create a Context with a Timeout
	// This says: "If Python doesn't answer in 10 seconds, hang up."
	// This prevents your Go server from freezing if Python crashes.
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// 2. Build the Request Object (The Envelope)
	req := &pb.TileRequest{
		RequestId: id,
		Width:     256,
		Height:    256,
		ImageData: imageData,
		ModelType: "cloud_mask", // We tell Python which model to use
	}

	// 3. Send it!
	return c.service.AnalyzeTile(ctx, req)
}