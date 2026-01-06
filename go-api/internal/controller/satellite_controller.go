package controller

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"sat-vis/internal/grpc_client"
	"sat-vis/internal/service"
	"sat-vis/internal/storage"
)

// SatelliteController handles HTTP requests related to satellite image processing.
// It manages the flow of uploading raw images, tiling them, sending them to the Python worker
// for analysis, stitching the results, and uploading the final mask.
type SatelliteController struct {
	tiler      service.TilerService
	grpcClient *grpc_client.TileClient
	storage *storage.StorageClient
}

// NewSatelliteController creates a new instance of SatelliteController with the given dependencies.
func NewSatelliteController(tiler service.TilerService, client *grpc_client.TileClient, storage *storage.StorageClient) *SatelliteController {
	return &SatelliteController{
		tiler:      tiler,
		grpcClient: client,
		storage: storage,
	}
}

// UploadAndAnalyze is the main HTTP handler for processing satellite images.
// It performs the following steps:
// 1. Receives the image file from the request.
// 2. Uploads the raw image to MinIO for backup.
// 3. Processes the image into tiles using the TilerService.
// 4. Sends each tile to the Python gRPC worker for cloud masking.
// 5. Stitches the resulting masks back into a single image.
// 6. Uploads the final mask image to MinIO.
// 7. Returns a JSON response with a presigned download URL for the result.
func (c *SatelliteController) UploadAndAnalyze(ctx *gin.Context) {
	fileHeader, err := ctx.FormFile("image")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "No image file provided"})
		return
	}

	// Generate a unique ID for this job
	jobID := fmt.Sprintf("job_%d", time.Now().Unix())
	inputFilename := fmt.Sprintf("%s_%s", jobID, fileHeader.Filename)
	
	// Open the uploaded file
	file, err := fileHeader.Open()
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open file"})
		return
	}
	defer file.Close()

	// --- STEP 1: Backup Raw Image to MinIO ---
	// We read the file size for MinIO
	fmt.Printf("[CONTROLLER] Uploading raw image to MinIO: %s\n", inputFilename)
	// Note: We use "bucket_input" from config, but here we hardcode the name logic for simplicity.
	// Ideally, expose the bucket name via a getter in StorageClient.
	err = c.storage.UploadStream(ctx, "satellite-raw", inputFilename, file, fileHeader.Size, fileHeader.Header.Get("Content-Type"))
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to upload to storage: " + err.Error()})
		return
	}

	// --- STEP 2: Rewind & Process ---
	// Since we read the file stream to upload it, the pointer is at the end.
	// We need to seek back to the start to read it again for processing.
	file.Seek(0, 0)

	tiles, bounds, err := c.tiler.ProcessImage(file)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Tiling failed: " + err.Error()})
		return
	}

	// --- STEP 3: Analyze & Stitch (Logic remains same) ---
	finalMask := image.NewRGBA(bounds)
	
	processedCount := 0
	errorCount := 0  // <--- Track errors

	for i, tile := range tiles {
		reqID := fmt.Sprintf("%s-tile-%d", jobID, i)
		
		resp, err := c.grpcClient.AnalyzeTile(context.Background(), reqID, tile.Data)
		
		// ERROR HANDLING LOGIC
		if err != nil {
			fmt.Printf("❌ Tile %d RPC Failed: %v\n", i, err)
			errorCount++
			continue // Skip this tile
		}
		
		if !resp.Success {
			fmt.Printf("❌ Tile %d Python Error: %s\n", i, resp.ErrorMessage)
			errorCount++
			continue // Skip this tile
		}

		// If success, draw the mask
		maskTile, err := png.Decode(bytes.NewReader(resp.ResultData))
		if err != nil {
			fmt.Printf("❌ Tile %d Decode Failed: %v\n", i, err)
			errorCount++
			continue
		}

		destRect := image.Rect(tile.X, tile.Y, tile.X+256, tile.Y+256)
		draw.Draw(finalMask, destRect, maskTile, image.Point{0, 0}, draw.Src)
		processedCount++
	}

	// --- CRITICAL CHECK: Did too many tiles fail? ---
	// If more than 10% failed, or if we processed nothing, abort.
	if processedCount == 0 || errorCount > (len(tiles)/10) {
		ctx.JSON(http.StatusBadGateway, gin.H{
			"error": fmt.Sprintf("Processing failed. Errors: %d, Success: %d. Check Python logs.", errorCount, processedCount),
		})
		return // Stop here! Do not save the blank image.
	}

	// --- STEP 4: Upload Result to MinIO ---
	// Encode result to Memory Buffer
	var outputBuf bytes.Buffer
	err = png.Encode(&outputBuf, finalMask)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to encode result"})
		return
	}

	outputFilename := fmt.Sprintf("mask_%s.png", jobID)
	fmt.Printf("[CONTROLLER] Uploading result mask to MinIO: %s\n", outputFilename)
	
	err = c.storage.UploadStream(ctx, "satellite-masks", outputFilename, &outputBuf, int64(outputBuf.Len()), "image/png")
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save result to storage"})
		return
	}

	// --- STEP 5: Generate Download Link ---
	downloadURL, _ := c.storage.GetPresignedURL(ctx, "satellite-masks", outputFilename)

	ctx.JSON(http.StatusOK, gin.H{
		"message":      "Processing Complete",
		"job_id":       jobID,
		"tiles_count":  processedCount,
		"result_url":   downloadURL, // <--- User gets a clickable link!
	})
}