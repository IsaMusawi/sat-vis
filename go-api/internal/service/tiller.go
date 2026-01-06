package service

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg" // Support decoding JPEGs
	"image/png"    // Support encoding/decoding PNGs
	"mime/multipart"

	"sat-vis/internal/config"

	// Import the draw package to ensure sub-imaging works correctly
	_ "golang.org/x/image/draw"
)

// TilerService defines the interface for splitting images into smaller tiles.
type TilerService interface {
	// ProcessImage decodes an uploaded file and splits it into a grid of tiles.
	// It returns a slice of Tile objects, the bounds of the original image, and any error encountered.
	ProcessImage(file multipart.File) ([]Tile, image.Rectangle, error)
}

type tilerServiceImpl struct {
	cfg *config.Config
}

// Tile represents a single square chunk of the original image.
type Tile struct {
	// Data contains the PNG-encoded bytes of the tile.
	Data []byte
	// X is the top-left X coordinate of the tile in the original image.
	X    int
	// Y is the top-left Y coordinate of the tile in the original image.
	Y    int
}

func NewTilerService(cfg *config.Config) TilerService {
	return &tilerServiceImpl{cfg: cfg}
}

// isTileEmpty checks if an image is fully transparent or black.
// It uses a sampling optimization to avoid iterating over every single pixel.
func isTileEmpty(img image.Image) bool {
	bounds := img.Bounds()
	// Optimization: Just check the center and a few points. 
	// If you want 100% accuracy, loop all pixels (slower).
	// Here we loop a sampling of pixels to be fast.
	for y := bounds.Min.Y; y < bounds.Max.Y; y += 10 {
		for x := bounds.Min.X; x < bounds.Max.X; x += 10 {
			_, _, _, a := img.At(x, y).RGBA()
			if a > 0 {
				return false // Found a visible pixel!
			}
		}
	}
	return true
}

// ProcessImage implements the TilerService interface.
// It splits the input image into fixed-size tiles (default 256x256).
// It handles edge cases by padding partial tiles with black.
// It also applies a "stride fix" by creating a fresh buffer for each tile to avoid memory layout distortion.
func (s *tilerServiceImpl) ProcessImage(file multipart.File) ([]Tile, image.Rectangle, error) {
	tileSize := s.cfg.Image.TileSize
	if tileSize <= 0 {
		tileSize = 256
	}

	srcImg, _, err := image.Decode(file)
	if err != nil {
		return nil, image.Rectangle{}, fmt.Errorf("failed to decode: %v", err)
	}

	bounds := srcImg.Bounds()
	var tiles []Tile
	
	fmt.Printf("[TILER] Processing Image: %dx%d\n", bounds.Dx(), bounds.Dy())

	// We create a Black Uniform image once to use for background padding
	black := &image.Uniform{color.RGBA{0, 0, 0, 255}}

	for y := bounds.Min.Y; y < bounds.Max.Y; y += tileSize {
		for x := bounds.Min.X; x < bounds.Max.X; x += tileSize {
			// 1. Define Boundaries
			x2 := min(x+tileSize, bounds.Max.X)
			y2 := min(y+tileSize, bounds.Max.Y)
			rect := image.Rect(x, y, x2, y2)

			// 2. Get the Data from Original Image
			type subImager interface {
				SubImage(r image.Rectangle) image.Image
			}
			simg, _ := srcImg.(subImager)
			virtualTile := simg.SubImage(rect)

			// 3. OPTIMIZATION: Check if this tile is just empty space
			if isTileEmpty(virtualTile) {
				continue // SKIP processing this tile!
			}

			// 4. THE STRIDE FIX: Create a clean 256x256 canvas
			// This removes the "Zig-Zag" distortion by resetting memory layout.
			cleanTile := image.NewRGBA(image.Rect(0, 0, tileSize, tileSize))
			
			// Fill with Black first (Safe background)
			draw.Draw(cleanTile, cleanTile.Bounds(), black, image.Point{}, draw.Src)

			// Copy the image data on top
			// The drawRect defines WHERE in the clean tile we paste the data (Top-Left: 0,0)
			drawRect := image.Rect(0, 0, x2-x, y2-y)
			draw.Draw(cleanTile, drawRect, virtualTile, rect.Min, draw.Over)

			// 5. Encode
			var buf bytes.Buffer
			png.Encode(&buf, cleanTile)

			tiles = append(tiles, Tile{
				Data: buf.Bytes(),
				X:    x,
				Y:    y,
			})
		}
	}

	return tiles, bounds, nil
}

// func (s *tilerServiceImpl) ProcessImage(file multipart.File) ([]Tile, image.Rectangle, error) {
// 	// --- NEW SAFETY CHECK ---
//     tileSize := s.cfg.Image.TileSize
//     if tileSize <= 0 {
//         // Fallback to 256 if config failed, or return error
//         fmt.Println("[TILER] ⚠️ Config TileSize is 0! Defaulting to 256.")
//         tileSize = 256
//     }
// 	// 1. Decode the Uploaded Image
// 	srcImg, format, err := image.Decode(file)
// 	if err != nil {
// 		return nil, image.Rectangle{}, fmt.Errorf("failed to decode image: %v", err)
// 	}
// 	fmt.Printf("[TILER] Decoded image format: %s. Bounds: %v\n", format, srcImg.Bounds())

// 	bounds := srcImg.Bounds()
	
// 	// FIX: Use the actual START coordinates of the image, not hardcoded 0
// 	minX := bounds.Min.X
// 	minY := bounds.Min.Y
// 	maxX := bounds.Max.X
// 	maxY := bounds.Max.Y

// 	var tiles []Tile

// 	// 2. The Tiling Loops (Respecting Bounds)
// 	for y := minY; y < maxY; y += tileSize {
// 		for x := minX; x < maxX; x += tileSize {
			
// 			// Calculate crop rectangle relative to the loop
// 			x2 := min(x+tileSize, maxX)
// 			y2 := min(y+tileSize, maxY)
			
// 			// Create the rectangle for this specific tile
// 			subRect := image.Rect(x, y, x2, y2)
			
// 			// INTERSECTION CHECK:
// 			// Ensure the request is actually inside the image
// 			intersect := subRect.Intersect(bounds)
// 			if intersect.Empty() {
// 				fmt.Printf("[TILER] ⚠️ Skipping empty tile at %v\n", subRect)
// 				continue
// 			}

// 			// 3. SubImage Extraction
// 			type subImager interface {
// 				SubImage(r image.Rectangle) image.Image
// 			}

// 			simg, ok := srcImg.(subImager)
// 			if !ok {
// 				return nil, image.Rectangle{}, fmt.Errorf("image format does not support tiling")
// 			}
			
// 			// Extract the tile
// 			tileImg := simg.SubImage(intersect)

// 			// 4. Encode to Bytes
// 			var buf bytes.Buffer
// 			if err := png.Encode(&buf, tileImg); err != nil {
// 				// Debug log to help identify which tile failed
// 				fmt.Printf("[TILER] ❌ Failed to encode tile at %v (Size: %dx%d)\n", 
// 					intersect, intersect.Dx(), intersect.Dy())
// 				return nil, image.Rectangle{}, err
// 			}
			
// 			tiles = append(tiles, Tile{
// 				Data: buf.Bytes(),
// 				X:    x,
// 				Y:y,
// 			})
// 		}
// 	}

// 	if len(tiles) == 0 {
// 		return nil, image.Rectangle{}, fmt.Errorf("no valid tiles generated from image")
// 	}

// 	return tiles, bounds, nil
// }

// func min(a, b int) int {
// 	if a < b {
// 		return a
// 	}
// 	return b
// }