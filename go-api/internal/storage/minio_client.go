package storage

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/url"
	"time"

	"sat-vis/internal/config"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type StorageClient struct {
	client *minio.Client
	cfg    *config.Config
}

// NewStorageClient initializes the connection
func NewStorageClient(cfg *config.Config) (*StorageClient, error) {
	minioClient, err := minio.New(cfg.Storage.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.Storage.AccessKey, cfg.Storage.SecretKey, ""),
		Secure: cfg.Storage.UseSSL,
	})
	if err != nil {
		return nil, err
	}

	// Auto-Create Buckets if they don't exist
	ctx := context.Background()
	buckets := []string{cfg.Storage.BucketInput, cfg.Storage.BucketOutput}
	
	for _, b := range buckets {
		exists, err := minioClient.BucketExists(ctx, b)
		if err != nil {
			return nil, fmt.Errorf("failed to check bucket %s: %v", b, err)
		}
		if !exists {
			log.Printf("[MINIO] Creating bucket: %s", b)
			err = minioClient.MakeBucket(ctx, b, minio.MakeBucketOptions{})
			if err != nil {
				return nil, fmt.Errorf("failed to create bucket: %v", err)
			}
		}
	}

	return &StorageClient{client: minioClient, cfg: cfg}, nil
}

// UploadStream uploads data directly from a stream (efficient RAM usage)
func (s *StorageClient) UploadStream(ctx context.Context, bucketName, objectName string, reader io.Reader, size int64, contentType string) error {
	_, err := s.client.PutObject(ctx, bucketName, objectName, reader, size, minio.PutObjectOptions{
		ContentType: contentType,
	})
	return err
}

// GetPresignedURL generates a temporary link for the user to download the result
func (s *StorageClient) GetPresignedURL(ctx context.Context, bucketName, objectName string) (string, error) {
	// Set link expiry to 1 hour
	reqParams := make(url.Values)
	presignedURL, err := s.client.PresignedGetObject(ctx, bucketName, objectName, time.Hour, reqParams)
	if err != nil {
		return "", err
	}
	return presignedURL.String(), nil
}