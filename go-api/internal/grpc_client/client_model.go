package grpc_client

import (
	"google.golang.org/grpc"

	pb "sat-vis/pkg/pb"
)


type TileClient struct {
	service pb.ImageAnalyzerClient
    conn *grpc.ClientConn	
}